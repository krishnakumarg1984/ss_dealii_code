// Include order: base-lac-grid-dofs-fe-numerics

// for assembling the matrix using quadrature on each cell
#include <deal.II/base/quadrature_lib.h>


// suppress unwanted output from the linear solvers
#include <deal.II/base/logstream.h>


// for the treatment of boundary values
#include <deal.II/base/function.h>


// for boundary values? (an overkill, as per a Github issue filed in Jan 2020)
#include <deal.II/base/function_lib.h>


// for using the parallel HDF5 DataOut binding
#include <deal.II/base/hdf5.h>


// to use a tensorial coefficient that may have a spatial dependence, the following include file provides the TensorFunction class
#include <deal.II/base/tensor_function.h>


// What is the below header file used for? (step-8)
#include <deal.II/base/tensor.h>


// to ensure that objects are not deleted while they are still in use, deal.II has the SmartPointer helper class declared in the following file
#include <deal.II/base/smartpointer.h>


// to use a ConvergenceTable that collects all important data during a run and prints it at the end as a table
#include <deal.II/base/convergence_table.h>


// What is the below header file used for? (steps 11,13)
#include <deal.II/base/table_handler.h>


// What is the below header file used for? (step 15, 26)
#include <deal.II/base/utilities.h>


/*----------------------------------------------------------------------------*/

// for the linear algebra employed to solve the system of equations arising from the finite element discretization of the underlying equation
#include <deal.II/lac/vector.h>
// or
#include <deal.II/lac/block_vector.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/sparse_matrix.h>
// or
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/lac/precondition_block.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>


// when using locally refined grids, we will get so-called hanging nodes. However, the standard finite element methods assumes that the discrete solution spaces be continuous, so we need to make sure that the degrees of freedom on hanging nodes conform to some constraints such that the global solution is continuous. We are also going to store the boundary conditions in this object. The following file contains a class which is used to handle these constraints:
#include <deal.II/lac/affine_constraints.h>


#include <deal.II/lac/sparse_ilu.h>

/*----------------------------------------------------------------------------*/

// The most fundamental class in the library is the Triangulation class, which is declared here
#include <deal.II/grid/tria.h>


// the following two includes are for loops over cells and/or faces
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>


// here are some functions to generate standard grids
#include <deal.II/grid/grid_generator.h>


// output of grids in various graphics formats
#include <deal.II/grid/grid_out.h>


// contains the class to read a triangulation from disk
#include <deal.II/grid/grid_in.h>


// contains facilities for describing circular and other boundary shapes
#include <deal.II/grid/manifold_lib.h>


// In order to refine our grids locally, we need a function that decides which cells to flag for refinement or coarsening based on the computed error indicators. This function is defined here
#include <deal.II/grid/grid_refinement.h>

/*----------------------------------------------------------------------------*/

// for the association of degrees of freedom ("DoF"s) to vertices, lines, and cells
#include <deal.II/dofs/dof_handler.h>


// In the following file, several tools for manipulating degrees of freedom can be found (eg for the creation of sparsity patterns of sparse matrices)
#include <deal.II/dofs/dof_tools.h>


// special algorithms to renumber degrees of freedom are declared here
#include <deal.II/dofs/dof_renumbering.h>


// contains classes needed for loops to get the information about the degrees of freedom local to a cell
#include <deal.II/dofs/dof_accessor.h>

/*----------------------------------------------------------------------------*/

// contains description of the bilinear finite element, including the facts that it has one degree of freedom on each vertex of the triangulation, but none on faces and none in the interior of the cells. (In fact, the file contains the description of Lagrange elements in general, i.e. also the quadratic, cubic, etc versions, and not only for 2d but also 1d and 3d.)
#include <deal.II/fe/fe_q.h>


// for assembling the matrix using quadrature on each cell
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>


// discontinuous Galerkin element
#include <deal.II/fe/fe_dgq.h>


// Raviart-Thomas element
#include <deal.II/fe/fe_raviart_thomas.h>


// Even if not solving a PDE, we'll need to use a dummy finite element with zero
// degrees of freedom provided by the FE_Nothing class from the following file
#include <deal.II/fe/fe_nothing.h>


// The MappingQ class is for polynomial mappings of arbitrary order, and is declared here
#include <deal.II/fe/mapping_q.h>


// What is the below header file used for? (step-11)
#include <deal.II/fe/mapping_q1.h>


// The following is needed for FEInterfaceValues to compute integrals on interfaces
#include <deal.II/fe/fe_interface_values.h>


/*----------------------------------------------------------------------------*/

// for the treatment of boundary values (also has the integrate_difference() function)
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>


// This is for output to a file
#include <deal.II/numerics/data_out.h>


// need a simple way to compute the refinement indicators based on some error estimate. While adaptivity is quite problem-specific, the error indicator in the following file often yields quite nicely adapted grids for a wide class of problems
#include <deal.II/numerics/error_estimator.h>


// This header is needed to use gradients as the refinement indicator
/* #include <deal.II/numerics/error_estimator.h> */


// This file defines the SolutionTransfer class for transferring the solution
// from an old mesh to the new (eg. adaptive mesh refinements b/w Newton iterations)
#include <deal.II/numerics/solution_transfer.h>


/*----------------------------------------------------------------------------*/


#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>

