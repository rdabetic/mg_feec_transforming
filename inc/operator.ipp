// =====================================================================
//                         TEMPLATE IMPLEMENTATION
// =====================================================================

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::WhitneyFormOperator(
    std::shared_ptr<Mesh> mesh,
    const BCond           bc,
    const OperatorMode    mode,
    const MassLumping     ml)
    : mfem::Operator(0),
      mesh_ptr_(std::move(mesh)),
      bc_(bc),
      opmode_(mode),
      ml_(ml)
{
    form_to_block_.fill(-1);

    // 1. Initialize Spaces for each active form degree
    (assembleSpace(ACTIVE_FORMS), ...);
    (assembleExteriorDerivative(ACTIVE_FORMS), ...);

    // 2. Setup MFEM Block Offsets
    setupBlockStructure();

    // 3. Assemble Hodge Stars (Mass Lumping)
    (assembleGalerkinMass(ACTIVE_FORMS), ...);

    if (ml_ == MassLumping::Barycentric)
        assembleGeometricLumpedMasses();

    // 4. Compute Topological Operators
    (computeDecOperators(ACTIVE_FORMS), ...);

    this->height = block_offsets_.Last();
    this->width  = block_offsets_.Last();

    if (bc_ == BCond::Essential)
        this->eliminateAllBoundaryConditions();

    setupTrivialHarmonics();
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    setupBlockStructure()
{
    constexpr std::array<unsigned int, sizeof...(ACTIVE_FORMS)> forms = {
        ACTIVE_FORMS...};
    block_offsets_.SetSize(forms.size() + 1);
    block_offsets_[0] = 0;

    for (size_t i = 0; i < forms.size(); ++i)
    {
        const unsigned int k  = forms[i];
        form_to_block_[k]     = static_cast<int>(i);
        block_offsets_[i + 1] = block_offsets_[i] + getBlockSize(k);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::assembleSpace(
    const unsigned int k)
{
    if (k > MAX_DIM || spaces_[k])
        return;

    const int                                      dim = mesh_ptr_->Dimension();
    std::shared_ptr<mfem::FiniteElementCollection> fec;

    switch (k)
    {
        case 0:
            fec = std::make_shared<mfem::H1_FECollection>(1, dim);
            break;
        case 1:
            fec = std::make_shared<mfem::ND_FECollection>(1, dim);
            break;
        case 2:
            if (dim == 3)
                fec = std::make_shared<mfem::RT_FECollection>(0, dim);
            else
                fec = std::make_shared<mfem::L2_FECollection>(
                    0, dim, mfem::BasisType::GaussLegendre,
                    mfem::FiniteElement::INTEGRAL);
            break;
        case 3:
            fec = std::make_shared<mfem::L2_FECollection>(
                0, dim, mfem::BasisType::GaussLegendre,
                mfem::FiniteElement::INTEGRAL);
            break;
    }

    if (fec)
    {
        fecs_[k]   = std::move(fec);
        spaces_[k] = std::make_shared<mfem::FiniteElementSpace>(
            mesh_ptr_.get(), fecs_[k].get());

        if (bc_ == BCond::Essential)
            buildEssentialDofMask(k);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    assembleExteriorDerivative(const unsigned int k)
{
    if (k >= mesh_ptr_->Dimension() || d_[k])
        return;

    if (!spaces_.at(k))
        assembleSpace(k);
    if (!spaces_.at(k + 1))
        assembleSpace(k + 1);

    switch (k)
    {
        case 0:
            d_[0] =
                assembleDiscreteGradient(spaces_[0].get(), spaces_[1].get());
            break;
        case 1:
            d_[1] = assembleDiscreteCurl(spaces_[1].get(), spaces_[2].get());
            break;
        case 2:
            d_[2] = assembleDiscreteDiv(spaces_[2].get(), spaces_[3].get());
            break;
        default:
            std::unreachable();
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    assembleGalerkinMass(const unsigned int k)
{
    if (k > MAX_DIM || !spaces_[k])
        return;

    mass_forms_[k] = std::make_shared<mfem::BilinearForm>(spaces_[k].get());
    mfem::ConstantCoefficient one(1.0);

    if (k == 0 || k == static_cast<unsigned int>(mesh_ptr_->Dimension()))
        mass_forms_[k]->AddDomainIntegrator(new mfem::MassIntegrator(one));
    else
        mass_forms_[k]->AddDomainIntegrator(
            new mfem::VectorFEMassIntegrator(one));

    mass_forms_[k]->Assemble();
    mass_forms_[k]->Finalize();

    if (ml_ == MassLumping::RowSum)
    {
        lumped_mass_vecs_[k].SetSize(mass_forms_[k]->NumRows());
        lumped_mass_vecs_[k] = 0.0;

        const auto& spmat = mass_forms_[k]->SpMat();
        const int* I      = spmat.GetI();
        const mfem::real_t* A = spmat.GetData();
        const int height = spmat.Height();

        for (int i = 0; i < height; ++i)
        {
            mfem::real_t row_sum = 0.0;
            for (int j = I[i]; j < I[i + 1]; ++j)
            {
                row_sum += std::abs(A[j]);
            }
            lumped_mass_vecs_[k][i] = row_sum;
        }
    }
    else if (ml_ == MassLumping::ScaledId)
    {
        lumped_mass_vecs_[k].SetSize(mass_forms_[k]->NumRows());
        lumped_mass_vecs_[k] =
            std::pow(mesh_ptr_->meshWidth(), mesh_ptr_->Dimension()) /
            std::pow(mesh_ptr_->meshWidth(), 2 * k);
    }
    // lumped_mass_vecs_[k] += std::pow(mesh_ptr_->meshWidth(),
    // mesh_ptr_->Dimension() + 1);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    buildEssentialDofMask(const int k)
{
    if (!spaces_[k])
        throw std::runtime_error(
            "buildEssentialDofMask: spaces_[k] not present!");

    essential_dof_[k] = std::make_shared<mfem::Array<int>>();

    spaces_[k]->GetBoundaryTrueDofs(*essential_dof_[k]);

    essential_dof_masks_[k] =
        std::make_shared<mfem::Array<int>>(getBlockSize(k));
    *essential_dof_masks_[k] = 0;

    for (int i = 0; i < essential_dof_[k]->Size(); ++i)
        (*essential_dof_masks_[k])[(*essential_dof_[k])[i]] = 1;
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    assembleGeometricLumpedMasses()
{
    DualMesh dual;
    // Try to convert from MassLumping to DualMesh
    if (ml_ == MassLumping::Barycentric)
        dual = DualMesh::Barycentric;
    else
        throw std::logic_error("Cannot convert to geometric dual!");

    const int dim = mesh_ptr_->Dimension();
    if (dim == 2)
        assembleGeometric2D(dual);
    else if (dim == 3)
        assembleGeometric3D(dual);
    else
        throw std::runtime_error("Only 2D and 3D meshes supported.");
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::assembleGeometric2D(
    const DualMesh dual)
{
    const int    nEdges = mesh_ptr_->GetNEdges(), nElems = mesh_ptr_->GetNE();
    mfem::Vector primalLen(nEdges), dualLen(nEdges), primalVol(nElems),
        dualVol(mesh_ptr_->GetNV());
    primalLen = 0.0;
    dualLen   = 0.0;
    dualVol   = 0.0;
    primalVol = 0.0;
    mfem::Array<int> edges, cor, vert;

    for (int i = 0; i < nElems; ++i)
    {
        primalVol(i)              = mesh_ptr_->GetElementVolume(i);
        const mfem::Vector center = mesh_ptr_->center(i, 2, dual);
        mesh_ptr_->GetElementEdges(i, edges, cor);
        for (int k = 0; k < edges.Size(); ++k)
        {
            const int eid = edges[k];
            mesh_ptr_->GetEdgeVertices(eid, vert);
            const mfem::Vector v0  = mesh_ptr_->vertToVec(vert[0]),
                               v1  = mesh_ptr_->vertToVec(vert[1]),
                               mid = mesh_ptr_->center(eid, 1, dual);
            primalLen(eid)         = v0.DistanceTo(v1);
            dualLen(eid) += center.DistanceTo(mid);
            dualVol(vert[0]) += triangleArea(v0, center, mid);
            dualVol(vert[1]) += triangleArea(v1, center, mid);
        }
    }
    lumped_mass_vecs_[0] = std::move(dualVol);
    lumped_mass_vecs_[1] = std::move(dualLen);
    lumped_mass_vecs_[1] /= primalLen;
    lumped_mass_vecs_[2] = std::move(primalVol);
    lumped_mass_vecs_[2].Reciprocal();
    lumped_mass_vecs_[2] /= 4.0;
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::assembleGeometric3D(
    const DualMesh dual)
{
    const int nEdges = mesh_ptr_->GetNEdges(), nFaces = mesh_ptr_->GetNFaces(),
              nElems = mesh_ptr_->GetNE();
    mfem::Vector pE(nEdges), dF(nEdges), pF(nFaces), dE(nFaces), pV(nElems),
        dV(mesh_ptr_->GetNV());
    pE = 0.0;
    dF = 0.0;
    pF = 0.0;
    dE = 0.0;
    pV = 0.0;
    dV = 0.0;
    mfem::Array<int> faces, edges, vert, cor;

    for (int i = 0; i < nElems; ++i)
    {
        pV(i)                   = mesh_ptr_->GetElementVolume(i);
        const mfem::Vector c_el = mesh_ptr_->center(i, 3, dual);
        mesh_ptr_->GetElementFaces(i, faces, cor);
        for (int kf = 0; kf < faces.Size(); ++kf)
        {
            const int          fid = faces[kf];
            const mfem::Vector c_f = mesh_ptr_->center(fid, 2, dual);
            pF(fid)                = mesh_ptr_->GetFaceArea(fid);
            dE(fid) += c_el.DistanceTo(c_f);
            mesh_ptr_->GetFaceEdges(fid, edges, cor);
            for (int ke = 0; ke < edges.Size(); ++ke)
            {
                const int eid = edges[ke];
                mesh_ptr_->GetEdgeVertices(eid, vert);
                const auto v0          = mesh_ptr_->vertToVec(vert[0]),
                           v1          = mesh_ptr_->vertToVec(vert[1]);
                const mfem::Vector c_e = mesh_ptr_->center(eid, 1, dual);
                pE(eid)                = v0.DistanceTo(v1);
                dF(eid) += triangleArea(c_e, c_f, c_el);
                dV(vert[0]) += tetrahedronVolume(v0, c_e, c_f, c_el);
                dV(vert[1]) += tetrahedronVolume(v1, c_e, c_f, c_el);
            }
        }
    }
    lumped_mass_vecs_[0] = std::move(dV);
    lumped_mass_vecs_[1] = std::move(dF);
    lumped_mass_vecs_[1] /= pE;
    lumped_mass_vecs_[2] = std::move(dE);
    lumped_mass_vecs_[2] /= pF;
    lumped_mass_vecs_[2] /= 4.0;
    lumped_mass_vecs_[3] = std::move(pV);
    lumped_mass_vecs_[3].Reciprocal();
    lumped_mass_vecs_[3] /= 36.0;
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::computeDecOperators(
    const unsigned int k)
{
    if (k < static_cast<unsigned int>(mesh_ptr_->Dimension()))
        assembleExteriorDerivative(k);
    if (k > 0)
    {
        assembleExteriorDerivative(k - 1);
        auto& d_prev = d_[k - 1];
        if (d_prev && !delta_[k])
        {
            delta_[k] =
                std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*d_prev));
            if (lumped_mass_vecs_[k].Size() > 0 &&
                lumped_mass_vecs_[k - 1].Size() > 0)
            {
                delta_[k]->ScaleColumns(lumped_mass_vecs_[k]);
                lumped_mass_vecs_[k - 1].Reciprocal();
                delta_[k]->ScaleRows(lumped_mass_vecs_[k - 1]);
                lumped_mass_vecs_[k - 1].Reciprocal();
            }
        }
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    eliminateBoundaryConditionsD(const unsigned int k)
{
    if (k >= MAX_DIM)
        return;
    if (!d_[k])
        throw std::invalid_argument(
            "eliminateBoundaryConditionsD: d_[k] == NULL");
    if (essential_dof_masks_[k] && essential_dof_masks_[k]->Size() > 0)
        d_[k]->EliminateCols(*essential_dof_masks_[k]);
    if (k + 1 <= MAX_DIM && essential_dof_[k + 1] &&
        essential_dof_[k + 1]->Size() > 0)
    {
        const auto& rows = *essential_dof_[k + 1];
        for (int i = 0; i < rows.Size(); ++i)
            d_[k]->EliminateRow(rows[i]);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    eliminateBoundaryConditionsDelta(const unsigned int k)
{
    if (k == 0 || k > MAX_DIM)
        return;
    if (!delta_[k])
        throw std::invalid_argument(
            "eliminateBoundaryConditionsD: delta_[k] == NULL");
    if (essential_dof_masks_[k] && essential_dof_masks_[k]->Size() > 0)
        delta_[k]->EliminateCols(*essential_dof_masks_[k]);
    if (k > 0 && essential_dof_[k - 1] && essential_dof_[k - 1]->Size() > 0)
    {
        const auto& rows = *essential_dof_[k - 1];
        for (int i = 0; i < rows.Size(); ++i)
            delta_[k]->EliminateRow(rows[i]);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    eliminateBoundaryConditionsMass(const unsigned int k)
{
    if (k > MAX_DIM)
        return;
    if (!mass_forms_[k])
        throw std::invalid_argument(
            "eliminateBoundaryConditionsD: !mass_forms_[k] == NULL");
    if (essential_dof_masks_[k] && essential_dof_masks_[k]->Size() > 0)
    {
        auto& spmat = mass_forms_[k]->SpMat();

        spmat.EliminateCols(*essential_dof_masks_[k]);

        const auto& rows = *essential_dof_[k];
        for (int i = 0; i < rows.Size(); ++i)
            spmat.EliminateRow(rows[i]);
    }

    // Does not eliminate the rows and columns???

    // if (essential_dof_masks_[k] && essential_dof_masks_[k]->Size() > 0)
    //     mass_forms_[k]->EliminateEssentialBC(*essential_dof_masks_[k],
    //                                          mfem::Operator::DiagonalPolicy::DIAG_ZERO);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    eliminateAllBoundaryConditions()
{
    constexpr std::array<unsigned int, sizeof...(ACTIVE_FORMS)> forms = {
        ACTIVE_FORMS...};
    for (unsigned k = 0; k < MAX_DIM + 1; ++k)
    {
        if (k < MAX_DIM && d_.at(k))
            eliminateBoundaryConditionsD(k);
        if (k > 0 && k < MAX_DIM + 1 && delta_.at(k))
            eliminateBoundaryConditionsDelta(k);
        if (mass_forms_.at(k))
            eliminateBoundaryConditionsMass(k);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::applyHodgeStar(
    mfem::BlockVector& u) const
{
    constexpr std::array<unsigned int, sizeof...(ACTIVE_FORMS)> forms = {
        ACTIVE_FORMS...};
    for (const unsigned int k : forms)
    {
        mfem::Vector& blk = u.GetBlock(form_to_block_[k]);
        blk *= lumped_mass_vecs_.at(k);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::applyInvHodgeStar(
    mfem::BlockVector& u) const
{
    constexpr std::array<unsigned int, sizeof...(ACTIVE_FORMS)> forms = {
        ACTIVE_FORMS...};
    for (const unsigned int k : forms)
    {
        mfem::Vector& blk = u.GetBlock(form_to_block_[k]);
        blk /= lumped_mass_vecs_.at(k);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
mfem::BlockVector
WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::createBlockVector() const
{
    return mfem::BlockVector(block_offsets_);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
unsigned int
WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::getBlockSize(
    const unsigned int k) const
{
    if (spaces_[k])
        return spaces_[k]->GetVSize();
    else
        throw std::runtime_error(
            "WhitneyFormOperator::getBlockSize: space not assembled!");
    // switch (k)
    // {
    //     case 0:
    //         return mesh_ptr_->GetNV();
    //     case 1:
    //         return mesh_ptr_->GetNEdges();
    //     case 2:
    //         return (mesh_ptr_->Dimension() == 2) ? mesh_ptr_->GetNE()
    //                                              : mesh_ptr_->GetNFaces();
    //     case 3:
    //         return mesh_ptr_->GetNE();
    //     default:
    //         return 0;
    // }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
mfem::Vector&
WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::getFormBlock(
    mfem::BlockVector& bv,
    const unsigned int k) const
{
    const int idx = form_to_block_[k];
    if (idx < 0)
        throw std::invalid_argument("Requested form degree is not active.");
    return bv.GetBlock(idx);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::Mult(
    const mfem::Vector& x,
    mfem::Vector&       y) const
{
    mfem::BlockVector bx(
        const_cast<mfem::real_t*>(x.GetData()), block_offsets_);
    mfem::BlockVector by(y.GetData(), block_offsets_);
    if (opmode_ == OperatorMode::Galerkin)
        applyFemOp(bx, by);
    else
        applyDecOp(bx, by);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::eliminateBC(
    mfem::Vector& x) const
{
    if (bc_ != BCond::Essential)
        throw std::runtime_error(
            "WhitneyFormOperator::EliminateBC: called with natural bc.");

    mfem::BlockVector x_(x.GetData(), block_offsets_);

    constexpr std::array<unsigned int, sizeof...(ACTIVE_FORMS)> forms = {
        ACTIVE_FORMS...};
    for (const unsigned int k : forms)
    {
        mfem::Vector& blk = x_.GetBlock(form_to_block_[k]);
        blk.SetSubVector(*essential_dof_[k], 0.0);
    }
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
mfem::BlockVector
WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::getRandomVector(
    int seed) const
{
    mfem::BlockVector r(block_offsets_);
    r.Randomize(seed);

    if (bc_ == BCond::Essential)
        eliminateBC(r);

    return r;
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
std::shared_ptr<DerivedOperator>
WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::createRefined() const
{
    auto new_mesh = std::make_shared<Mesh>(*this->mesh_ptr_);
    new_mesh->refine();

    return std::make_shared<DerivedOperator>(
        new_mesh, this->bc_, this->opmode_, this->ml_);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    eliminateTrivialSolutionHarmonicsBlock(mfem::BlockVector& u) const
{
    if (harmonic_form_degree_ < 0)
        return;

    mfem::Vector& uk = u.GetBlock(harmonic_form_degree_);

    // Orthogonal projection: u = u - (u, c)_M / (c, c)_M * c
    // Note: (u, c)_M = u^T * (M * c) = uk * Mc_
    const mfem::real_t alpha = (uk * Mc_) / cMc_;

    uk.Add(-alpha, c_);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    eliminateTrivialSolutionHarmonics(mfem::Vector& u) const
{
    mfem::BlockVector bu(u.GetData(), block_offsets_);
    eliminateTrivialSolutionHarmonicsBlock(bu);
}

template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
void WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>::
    setupTrivialHarmonics()
{
    // Identify nullspace block: Natural -> 0-forms, Essential -> dim-forms
    const int k = (bc_ == BCond::Natural) ? 0 : mesh_ptr_->Dimension();

    // Check if k is in the active forms
    if (((k == ACTIVE_FORMS) || ...))
    {
        harmonic_form_degree_ = k;
    }
    else  // set to ignore (anything < 0)
    {
        harmonic_form_degree_ = -1;
        return;
    }

    const int size = this->getBlockSize(k);

    c_.SetSize(size);

    mfem::ConstantCoefficient one(1.0);
    mfem::GridFunction        gf(this->spaces_[k].get());
    gf.ProjectCoefficient(one);
    c_ = gf;

    // TODO: Delete
    // if(bc_ == BCond::Essential)
    //     this->eliminateBC(c_);

    Mc_.SetSize(size);

    if (this->opmode_ == OperatorMode::DEC)
    {
        Mc_ = c_;
        Mc_ *= this->getLumpedMass(k);
    }
    else
    {
        this->getMassMatrix(k)->Mult(c_, Mc_);
    }

    cMc_ = c_ * Mc_;  // Inner product
}