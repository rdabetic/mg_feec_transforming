#ifndef INCIDENCE_H
#define INCIDENCE_H

#include "mfem.hpp"

std::unique_ptr<mfem::SparseMatrix> assembleDiscreteGradient(
    mfem::FiniteElementSpace* h1,
    mfem::FiniteElementSpace* hcurl);

std::unique_ptr<mfem::SparseMatrix> assembleDiscreteCurl(
    mfem::FiniteElementSpace* hcurl,
    mfem::FiniteElementSpace* hdiv);

std::unique_ptr<mfem::SparseMatrix> assembleDiscreteDiv(
    mfem::FiniteElementSpace* hdiv,
    mfem::FiniteElementSpace* l2);

#endif
