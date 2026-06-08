// =======================================================================
//                    DEC OPERATOR WRAPPER IMPLEMENTATION
// =======================================================================

template <typename OpType, unsigned int... FORMS>
DecWrapper<OpType, FORMS...>::DecWrapper(std::shared_ptr<const OpType> op)
    : mfem::Operator(op->Height(), op->Width()), op_(std::move(op))
{
    offsets_.SetSize(sizeof...(FORMS) + 1);
    offsets_[0] = 0;

    unsigned int sizes[] = {op_->getBlockSize(FORMS)...};
    for (size_t i = 0; i < sizeof...(FORMS); ++i)
        offsets_[i + 1] = offsets_[i] + sizes[i];
}

template <typename OpType, unsigned int... FORMS>
void DecWrapper<OpType, FORMS...>::Mult(const mfem::Vector& x, mfem::Vector& y)
    const
{
    mfem::BlockVector bx(const_cast<mfem::real_t*>(x.GetData()), offsets_);
    mfem::BlockVector by(y.GetData(), offsets_);
    op_->applyDecOp(bx, by);
}

// =======================================================================
//                    MULTIGRID SOLVER IMPLEMENTATION
// =======================================================================

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::Multigrid(
    const OperatorMode mode)
    : mfem::Solver(0),
      mg_mode_(mode),
      cycle_type_(CycleType::VCYCLE),
      pre_smooth_iters_(1),
      post_smooth_iters_(1)
{
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::
    createCoarseSolver()
{
    if (operators_.size() == 0)
        throw std::logic_error(
            "createCoarseSolver: Coarse operator undefined!");

    auto coarse_mat = operators_[0]->assembleDec();
    if (!coarse_mat)
    {
        throw std::runtime_error(
            "Coarsest operator failed to assemble the DEC matrix.");
    }

    coarse_solver_ = std::make_shared<EigenSparseQRSolver>(coarse_mat);
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
template <typename... SMOOTHER_ARGS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::addLevel(
    std::shared_ptr<DerivedOperator> fine_op,
    SMOOTHER_ARGS... args)
{
    if (!fine_op)
        throw std::invalid_argument("Cannot add a null operator.");

    operators_.push_back(fine_op);

    auto dec_wrapper = std::make_unique<WrapperType>(fine_op);
    dec_wrappers_.push_back(std::move(dec_wrapper));

    if (operators_.size() == 1)
    {
        createCoarseSolver();
        smoothers_.push_back(nullptr);
    }
    else
    {
        auto smoother = std::make_unique<SmootherType>(
            fine_op, std::forward<SMOOTHER_ARGS>(args)...);

        smoother->iterative_mode = true;
        smoothers_.push_back(std::move(smoother));

        const size_t L = operators_.size() - 1;
        auto P = buildBlockProlongation(*operators_[L - 1], *operators_[L]);
        prolongations_.push_back(std::move(P));
    }

    this->height = fine_op->Height();
    this->width  = fine_op->Width();
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
template <typename... SMOOTHER_ARGS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::addRefinedLevel(
    SMOOTHER_ARGS... args)
{
    if (operators_.empty())
    {
        throw std::logic_error(
            "Cannot refine an empty hierarchy. Add a coarse level first.");
    }

    // Explicit cast needed if createRefined returns base WhitneyFormOperator
    auto fine_op = operators_.back()->createRefined();
    if (!fine_op)
        throw std::runtime_error("fine_op == nullptr.");

    addLevel(fine_op, std::forward<SMOOTHER_ARGS>(args)...);
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
template <typename... OPERATOR_ARGS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::addCoarseLevel(
    std::shared_ptr<mfem::Solver> coarse_solver,
    OPERATOR_ARGS... args)
{
    if (!operators_.empty())
        throw std::logic_error("initCoarseLevel: Coarse level already exists.");

    auto op =
        std::make_shared<DerivedOperator>(std::forward<OPERATOR_ARGS>(args)...);

    operators_.push_back(op);

    auto dec_wrapper = std::make_unique<WrapperType>(op);
    dec_wrappers_.push_back(std::move(dec_wrapper));

    smoothers_.push_back(nullptr);

    if (coarse_solver)
        coarse_solver_ = coarse_solver;
    else
        createCoarseSolver();

    this->height = op->Height();
    this->width  = op->Width();
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::Mult(
    const mfem::Vector& b,
    mfem::Vector&       x) const
{
    if (operators_.empty())
        return;

    if (!iterative_mode)
        x = 0.0;

    auto&             op = operators_.back();
    mfem::BlockVector bx(x.GetData(), op->getBlockOffsets());

    mfem::Vector b_scaled(b);
    if (mg_mode_ == OperatorMode::Galerkin)
    {
        mfem::BlockVector b_blk(b_scaled.GetData(), op->getBlockOffsets());
        op->applyInvHodgeStar(b_blk);
    }

    mgCycle(static_cast<int>(operators_.size()) - 1, b_scaled, x);

    if (eliminate_trivial_harmonics_)
        op->eliminateTrivialSolutionHarmonics(bx);
    // op->eliminateBC(x);
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::MultTranspose(
    const mfem::Vector& b,
    mfem::Vector&       x) const
{
    // NOTE: The transpose multigrid cycle is not implemented here.
    //       This fallback simply reuses the forward Mult() action.
    //       That is only correct for a truly self-adjoint operator or when
    //       transpose application is not required.
        MFEM_ABORT("MultTranspose not implemented for Multigrid. "
                "This fallback is not mathematically correct for non-symmetric operators.");
    Mult(b, x);
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
void Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::mgCycle(
    const int           level,
    const mfem::Vector& b,
    mfem::Vector&       x) const
{
    if (level == 0)
    {
        coarse_solver_->Mult(b, x);
        // Just to make sure (due to rounding errors)
        operators_.at(0)->eliminateTrivialSolutionHarmonics(x);
	if(operators_.at(0)->getBCond() == BCond::Essential)
	  operators_.at(0)->eliminateBC(x);

        return;
    }

    for (int i = 0; i < pre_smooth_iters_; ++i)
        smoothers_[level]->Mult(b, x);

    mfem::Vector r(b.Size());
    dec_wrappers_[level]->Mult(x, r);
    r -= b;

    mfem::Vector r_c(operators_[level - 1]->Height());
    prolongations_[level - 1]->MultTranspose(r, r_c);

    mfem::Vector e_c(r_c.Size());
    e_c = 0.0;

    const int num_cycles = (cycle_type_ == CycleType::WCYCLE) ? 2 : 1;
    for (int c = 0; c < num_cycles; ++c)
        mgCycle(level - 1, r_c, e_c);

    prolongations_[level - 1]->AddMult(e_c, x, -1.0);

    for (int i = 0; i < post_smooth_iters_; ++i)
            smoothers_[level]->Mult(b, x);
}

template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
std::unique_ptr<mfem::BlockOperator>
Multigrid<DerivedOperator, SmootherType, ACTIVE_FORMS...>::
    buildBlockProlongation(
        const OperatorType& coarse_op,
        const OperatorType& fine_op) const
{
    constexpr std::array<unsigned int, sizeof...(ACTIVE_FORMS)> forms = {
        ACTIVE_FORMS...};

    mfem::Array<int> row_offsets(forms.size() + 1);
    mfem::Array<int> col_offsets(forms.size() + 1);
    row_offsets[0] = 0;
    col_offsets[0] = 0;

    for (size_t i = 0; i < forms.size(); ++i)
    {
        row_offsets[i + 1] = row_offsets[i] + fine_op.getBlockSize(forms[i]);
        col_offsets[i + 1] = col_offsets[i] + coarse_op.getBlockSize(forms[i]);
    }

    auto P = std::make_unique<mfem::BlockOperator>(row_offsets, col_offsets);
    P->owns_blocks = true;

    for (size_t i = 0; i < forms.size(); ++i)
    {
        const unsigned int k = forms[i];

        auto coarse_space = coarse_op.getSpace(k);
        auto fine_space   = fine_op.getSpace(k);

        if (!coarse_space || !fine_space)
        {
            throw std::runtime_error(
                "Spaces not initialized properly for multigrid transfer.");
        }

        const mfem::Vector& star_c = coarse_op.getLumpedMass(k);
        const mfem::Vector& star_f = fine_op.getLumpedMass(k);

        auto ess_c = coarse_op.getEssentialTrueDofs(k);
        auto ess_f = fine_op.getEssentialTrueDofs(k);

        auto* transfer = new TransferOperator(
            *coarse_space, *fine_space, star_c, star_f, ess_c, ess_f);

        P->SetBlock(static_cast<int>(i), static_cast<int>(i), transfer);
    }

    return P;
}
