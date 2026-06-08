#include "mfem.hpp"

inline mfem::real_t triangleArea(
    const mfem::Vector& a,
    const mfem::Vector& b,
    const mfem::Vector& c)
{
    const mfem::real_t s0 = b.DistanceTo(a), s1 = c.DistanceTo(b),
                       s2 = a.DistanceTo(c), s = (s0 + s1 + s2) / 2.;
    return std::sqrt(s * (s - s0) * (s - s1) * (s - s2));
}

inline mfem::real_t tetrahedronVolume(
    const mfem::Vector& x0,
    const mfem::Vector& x1,
    const mfem::Vector& x2,
    const mfem::Vector& x3)
{
    mfem::Vector d10 = x1, d20 = x2, d30 = x3;
    d10 -= x0;
    d20 -= x0, d30 -= x0;

    mfem::Vector cross(3);
    d20.cross3D(d30, cross);

    return std::abs(d10 * cross) / 6.;
}
