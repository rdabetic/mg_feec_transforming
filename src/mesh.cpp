#include "mesh.hpp"

#include <cmath>
#include <stdexcept>

Mesh::Mesh(const Mesh& mesh) : mfem::Mesh(mesh) { updateGeometricProperties(); }

Mesh::Mesh(mfem::Mesh&& mesh) : mfem::Mesh(std::move(mesh))
{
    updateGeometricProperties();
}

void Mesh::updateGeometricProperties()
{
    h_                   = 0.0;
    const int        dim = Dimension();
    mfem::Array<int> vert(2);

    for (int ei = 0; ei < GetNEdges(); ++ei)
    {
        GetEdgeVertices(ei, vert);
        const mfem::real_t* v1 = GetVertex(vert[0]);
        const mfem::real_t* v2 = GetVertex(vert[1]);

        mfem::real_t distSq = 0.0;
        for (int d = 0; d < dim; ++d)
        {
            const mfem::real_t diff = v1[d] - v2[d];
            distSq += diff * diff;
        }
        h_ = std::max(h_, std::sqrt(distSq));
    }
}

void Mesh::refine()
{
    UniformRefinement();
    updateGeometricProperties();
}

mfem::Vector Mesh::barycenter(const int id, const int k) const
{
    mfem::Array<int> vids;
    if (k == 0)
        vids.Append(id);
    else if (k == 1)
        GetEdgeVertices(id, vids);
    else if (k == 2)
        (Dimension() == 2) ? GetElementVertices(id, vids)
                           : GetFaceVertices(id, vids);
    else
        GetElementVertices(id, vids);

    const int    dim = Dimension();
    mfem::Vector c(dim);
    c = 0.0;

    for (int i = 0; i < vids.Size(); ++i)
    {
        const mfem::real_t* v = GetVertex(vids[i]);
        for (int d = 0; d < dim; ++d)
            c(d) += v[d];
    }
    c /= static_cast<mfem::real_t>(vids.Size());
    return c;
}

mfem::Vector Mesh::center(const int elemID, const int k, const DualMesh dual)
    const
{
    if (dual == DualMesh::Barycentric || k <= 1)
        return barycenter(elemID, k);

    throw std::runtime_error("Center not implemented for this dual type");
}

mfem::Vector Mesh::vertToVec(const int vid) const
{
    return mfem::Vector(const_cast<mfem::real_t*>(GetVertex(vid)), Dimension());
}

mfem::real_t Mesh::GetFaceArea(const int fid)
{
    // Get the transformation from the reference face to the physical face
    mfem::ElementTransformation* T = GetFaceTransformation(fid);

    // Get an integration rule suitable for the face's geometry and order
    // For linear meshes, even a low-order rule is exact.
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(T->GetGeometryType(), T->Order());

    mfem::real_t area = 0.0;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const mfem::IntegrationPoint& ip = ir.IntPoint(i);
        T->SetIntPoint(&ip);

        // T->Weight() returns the determinant of the Jacobian (the scaling
        // factor) at the specific integration point.
        area += ip.weight * T->Weight();
    }
    return area;
}
