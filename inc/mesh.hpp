#ifndef MESH_H
#define MESH_H

#include <algorithm>
#include <array>
#include <memory>

#include "incidence.hpp"
#include "mfem.hpp"

class Mesh : public mfem::Mesh
{
public:
    static constexpr int MAX_DIM = 3;

private:
    mfem::real_t h_ = 0.0;

    void updateGeometricProperties();

public:
    using mfem::Mesh::vertices;

    Mesh(const Mesh& mesh);
    Mesh(mfem::Mesh&& mesh);

    void refine();

    [[nodiscard]] mfem::Vector barycenter(const int id, const int k) const;
    [[nodiscard]] mfem::real_t meshWidth() const { return h_; }
    [[nodiscard]] mfem::Vector vertToVec(const int vid) const;

    // Not const (mfem's GetFaceTransformation() is not const)
    [[nodiscard]] mfem::real_t GetFaceArea(const int fid);
};

#endif
