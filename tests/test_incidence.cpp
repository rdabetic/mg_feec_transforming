#include <gtest/gtest.h>

#include "incidence.hpp"
#include "mfem.hpp"

// Helper to setup the de Rham sequence spaces
struct DeRhamSpaces
{
    std::unique_ptr<mfem::H1_FECollection> h1_fec;
    std::unique_ptr<mfem::ND_FECollection> hcurl_fec;
    std::unique_ptr<mfem::RT_FECollection> hdiv_fec;
    std::unique_ptr<mfem::L2_FECollection> l2_fec;

    std::unique_ptr<mfem::FiniteElementSpace> h1;
    std::unique_ptr<mfem::FiniteElementSpace> hcurl;
    std::unique_ptr<mfem::FiniteElementSpace> hdiv;
    std::unique_ptr<mfem::FiniteElementSpace> l2;

    DeRhamSpaces(mfem::Mesh* mesh, int dim)
    {
        h1_fec    = std::make_unique<mfem::H1_FECollection>(1, dim);
        hcurl_fec = std::make_unique<mfem::ND_FECollection>(1, dim);
        hdiv_fec  = std::make_unique<mfem::RT_FECollection>(0, dim);
        l2_fec    = std::make_unique<mfem::L2_FECollection>(0, dim);

        h1 = std::make_unique<mfem::FiniteElementSpace>(mesh, h1_fec.get());
        hcurl =
            std::make_unique<mfem::FiniteElementSpace>(mesh, hcurl_fec.get());
        hdiv = std::make_unique<mfem::FiniteElementSpace>(mesh, hdiv_fec.get());
        l2   = std::make_unique<mfem::FiniteElementSpace>(mesh, l2_fec.get());
    }
};

TEST(IncidenceTest, ComplexPropertyTria)
{
    const unsigned int n = 5, DIM = 2;
    mfem::Mesh         mesh =
        mfem::Mesh::MakeCartesian2D(n, n + 1, mfem::Element::TRIANGLE);
    DeRhamSpaces s(&mesh, DIM);

    const mfem::SparseMatrix d0 = *assembleDiscreteGradient(
                                 s.h1.get(), s.hcurl.get()),
                             // In 2D, Curl maps H(curl) to L2
        d1 = *assembleDiscreteCurl(s.hcurl.get(), s.l2.get());

    const std::unique_ptr<const mfem::SparseMatrix> d1d0 =
        std::unique_ptr<const mfem::SparseMatrix>(mfem::Mult(d1, d0));

    ASSERT_FLOAT_EQ(d1d0->MaxNorm(), 0.0);
    ASSERT_GT(d0.MaxNorm(), 0.0);
    ASSERT_GT(d1.MaxNorm(), 0.0);
}

TEST(IncidenceTest, ComplexPropertyQuad)
{
    const unsigned int n = 5, DIM = 2;
    mfem::Mesh         mesh =
        mfem::Mesh::MakeCartesian2D(n, n + 1, mfem::Element::QUADRILATERAL);
    DeRhamSpaces s(&mesh, DIM);

    const mfem::SparseMatrix d0 = *assembleDiscreteGradient(
                                 s.h1.get(), s.hcurl.get()),
                             d1 = *assembleDiscreteCurl(
                                 s.hcurl.get(), s.l2.get());

    const std::unique_ptr<const mfem::SparseMatrix> d1d0 =
        std::unique_ptr<const mfem::SparseMatrix>(mfem::Mult(d1, d0));

    ASSERT_FLOAT_EQ(d1d0->MaxNorm(), 0.0);
    ASSERT_GT(d0.MaxNorm(), 0.0);
    ASSERT_GT(d1.MaxNorm(), 0.0);
}

TEST(IncidenceTest, ComplexPropertyTets)
{
    const unsigned int n = 5, DIM = 3;
    mfem::Mesh         mesh = mfem::Mesh::MakeCartesian3D(
        n, n + 1, n + 2, mfem::Element::TETRAHEDRON);
    DeRhamSpaces s(&mesh, DIM);

    const mfem::SparseMatrix d0 = *assembleDiscreteGradient(
                                 s.h1.get(), s.hcurl.get()),
                             d1 = *assembleDiscreteCurl(
                                 s.hcurl.get(), s.hdiv.get()),
                             d2 =
                                 *assembleDiscreteDiv(s.hdiv.get(), s.l2.get());

    const std::unique_ptr<const mfem::SparseMatrix>
        d1d0 = std::unique_ptr<const mfem::SparseMatrix>(mfem::Mult(d1, d0)),
        d2d1 = std::unique_ptr<const mfem::SparseMatrix>(mfem::Mult(d2, d1));

    ASSERT_FLOAT_EQ(d1d0->MaxNorm(), 0.0);
    ASSERT_FLOAT_EQ(d2d1->MaxNorm(), 0.0);
    ASSERT_GT(d0.MaxNorm(), 0.0);
    ASSERT_GT(d1.MaxNorm(), 0.0);
    ASSERT_GT(d2.MaxNorm(), 0.0);
}

TEST(IncidenceTest, ComplexPropertyHex)
{
    const unsigned int n = 5, DIM = 3;
    mfem::Mesh         mesh =
        mfem::Mesh::MakeCartesian3D(n, n + 1, n + 2, mfem::Element::HEXAHEDRON);
    DeRhamSpaces s(&mesh, DIM);

    const mfem::SparseMatrix d0 = *assembleDiscreteGradient(
                                 s.h1.get(), s.hcurl.get()),
                             d1 = *assembleDiscreteCurl(
                                 s.hcurl.get(), s.hdiv.get()),
                             d2 =
                                 *assembleDiscreteDiv(s.hdiv.get(), s.l2.get());

    const std::unique_ptr<const mfem::SparseMatrix>
        d1d0 = std::unique_ptr<const mfem::SparseMatrix>(mfem::Mult(d1, d0)),
        d2d1 = std::unique_ptr<const mfem::SparseMatrix>(mfem::Mult(d2, d1));

    ASSERT_FLOAT_EQ(d1d0->MaxNorm(), 0.0);
    ASSERT_FLOAT_EQ(d2d1->MaxNorm(), 0.0);
    ASSERT_GT(d0.MaxNorm(), 0.0);
    ASSERT_GT(d1.MaxNorm(), 0.0);
    ASSERT_GT(d2.MaxNorm(), 0.0);
}

TEST(IncidenceTest, CurlCurl2D)
{
    const unsigned int        n = 1, DIM = 2;
    mfem::ConstantCoefficient one(1.0);
    mfem::Mesh                mesh =
        mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::TRIANGLE);
    DeRhamSpaces s(&mesh, DIM);

    const int ne = mesh.GetNEdges(), nf = mesh.GetNE();

    const mfem::SparseMatrix d0 = *assembleDiscreteGradient(
                                 s.h1.get(), s.hcurl.get()),
                             d1 = *assembleDiscreteCurl(
                                 s.hcurl.get(), s.l2.get());

    ASSERT_EQ(d0.NumRows(), ne);
    ASSERT_EQ(d1.NumRows(), nf);
    ASSERT_EQ(d1.NumCols(), ne);

    auto mass_ = std::make_unique<mfem::BilinearForm>(s.l2.get());
    mass_->AddDomainIntegrator(new mfem::MassIntegrator(one));
    mass_->Assemble();
    mass_->Finalize();
    const mfem::SparseMatrix& mass = mass_->SpMat();

    auto curlcurl_ = std::make_unique<mfem::BilinearForm>(s.hcurl.get());
    curlcurl_->AddDomainIntegrator(new mfem::CurlCurlIntegrator(one));
    curlcurl_->Assemble();
    curlcurl_->Finalize();
    const mfem::SparseMatrix& curlcurl = curlcurl_->SpMat();

    std::unique_ptr<mfem::SparseMatrix> curlcurl_incidence_;
    {
        auto tmp = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(mass, d1));
        auto d1T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(d1));
        curlcurl_incidence_ =
            std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(*d1T, *tmp));
    }

    auto diff = std::unique_ptr<mfem::SparseMatrix>(
        mfem::Add(1.0, curlcurl, -1.0, *curlcurl_incidence_));
    ASSERT_LT(diff->MaxNorm() / curlcurl.MaxNorm(), 1e-10);
}

TEST(IncidenceTest, CurlCurl3D)
{
    const unsigned int        n = 5, DIM = 3;
    mfem::ConstantCoefficient one(1.0);
    mfem::Mesh                mesh = mfem::Mesh::MakeCartesian3D(
        n, n + 1, n + 2, mfem::Element::TETRAHEDRON);
    DeRhamSpaces s(&mesh, DIM);

    const int ne = mesh.GetNEdges(), nf = mesh.GetNFaces();

    const mfem::SparseMatrix d0 = *assembleDiscreteGradient(
                                 s.h1.get(), s.hcurl.get()),
                             d1 = *assembleDiscreteCurl(
                                 s.hcurl.get(), s.hdiv.get());

    ASSERT_EQ(d0.NumRows(), ne);
    ASSERT_EQ(d1.NumRows(), nf);
    ASSERT_EQ(d1.NumCols(), ne);

    auto mass_ = std::make_unique<mfem::BilinearForm>(s.hdiv.get());
    mass_->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
    mass_->Assemble();
    mass_->Finalize();
    const mfem::SparseMatrix& mass = mass_->SpMat();

    auto curlcurl_ = std::make_unique<mfem::BilinearForm>(s.hcurl.get());
    curlcurl_->AddDomainIntegrator(new mfem::CurlCurlIntegrator(one));
    curlcurl_->Assemble();
    curlcurl_->Finalize();
    const mfem::SparseMatrix& curlcurl = curlcurl_->SpMat();

    std::unique_ptr<mfem::SparseMatrix> curlcurl_incidence_;
    {
        auto tmp = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(mass, d1));
        auto d1T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(d1));
        curlcurl_incidence_ =
            std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(*d1T, *tmp));
    }

    auto diff = std::unique_ptr<mfem::SparseMatrix>(
        mfem::Add(1.0, curlcurl, -1.0, *curlcurl_incidence_));
    ASSERT_LT(diff->MaxNorm() / curlcurl.MaxNorm(), 1e-10);
}