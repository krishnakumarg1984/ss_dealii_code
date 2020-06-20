/*
 * Solid State Battery Diffusion Simulator
 * Copyright © 2020 Krishnakumar Gopalakrishnan, University College London
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
// #include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace SolidDiffusion
{
  using namespace dealii;

  template <int dim>
  class DiffusionEquation
  {
  public:
    DiffusionEquation();
    void run();

  private:
    void setup_system();
    void assemble_neumann_rhs();
    void solve_time_step();
    void output_results() const;
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints; // hanging_node_constraints

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;
    Vector<double> system_rhs_neumannbc;

    double       time;
    double       time_step;
    unsigned int timestep_number;

    const double theta;
  };


  double DiffusionCoefficient() // for now return a constant value; later on
                                // interpolate from previous two solutions
  {
    return 1.0;
  }


  // Constructor for the class DiffusionEquation
  template <int dim>
  DiffusionEquation<dim>::DiffusionEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time(0.0)
    , time_step(0.1) // 1 sec or 10 sec is okay as per MC
    , timestep_number(0)
    , theta(0.5)
  {}


  template <int dim>
  void DiffusionEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    system_rhs_neumannbc.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void DiffusionEquation<dim>::assemble_neumann_rhs()
  {
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_JxW_values);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0.;


        for (unsigned int face_number = 0;
             face_number < GeometryInfo<dim>::faces_per_cell;
             ++face_number)
          if (cell->face(face_number)->at_boundary() &&
              (cell->face(face_number)->boundary_id() == 0))
            // In 1D, boundary_id of left edge is 0 and right edge is 1
            {
              fe_face_values.reinit(cell, face_number);
              for (unsigned int q_point = 0; q_point < n_face_q_points;
                   ++q_point)
                {
                  double neumann_value = -0.001; // currently hard coded
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) +=
                      (neumann_value *                          // g(x_q)
                       fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                       fe_face_values.JxW(q_point));            // dx
                }
            }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs_neumannbc(local_dof_indices[i]) += cell_rhs(i);
      }
  } // namespace SolidDiffusion

  template <int dim>
  void DiffusionEquation<dim>::solve_time_step()
  {
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<>    cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }


  template <int dim>
  void DiffusionEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "x_Li");

    data_out.build_patches();

    const std::string filename = "solution_timestep_" +
                                 Utilities::int_to_string(timestep_number, 3) +
                                 ".vtu";
    // ".gnuplot";
    std::ofstream output(filename);
    data_out.write_vtu(output);
    // data_out.write_gnuplot(output);
  }


  template <int dim>
  void DiffusionEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                           const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.6,
                                                      0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement();
    setup_system();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute(solution);
  }


  template <int dim>
  void DiffusionEquation<dim>::run()
  {
    const unsigned int initial_global_refinement       = 3;
    const unsigned int n_adaptive_pre_refinement_steps = 6;

    GridGenerator::hyper_cube(triangulation); // In 1D, a hypercube is a line
    triangulation.refine_global(initial_global_refinement);

    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp; // for holding temporary RHS quantities

  start_time_iteration:

    tmp.reinit(solution.size());


    VectorTools::interpolate(dof_handler,
                             // Functions::ZeroFunction<dim>(),
                             Functions::ConstantFunction<dim>(0.5), // init x_Li
                             old_solution);

    solution = old_solution;

    output_results();

    while (time <= 10) // end time is hard-coded here for now
      {
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution); // MU^(n-1) into system_rhs

        laplace_matrix.vmult(tmp, old_solution); // AU^(n-1) into tmp
        system_rhs.add(-(1 - theta) * time_step * DiffusionCoefficient(), tmp);

        assemble_neumann_rhs();

        system_rhs += system_rhs_neumannbc;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step * DiffusionCoefficient(),
                          laplace_matrix);

        constraints.condense(system_matrix, system_rhs);

        solve_time_step();

        output_results();

        if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;
            tmp.reinit(solution.size());
            system_rhs_neumannbc.reinit(solution.size());
            std::cout << std::endl;
            goto start_time_iteration;
          }
        else if ((timestep_number > 0) && (timestep_number % 5 == 0))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            tmp.reinit(solution.size());
            system_rhs_neumannbc.reinit(solution.size());
          }

        old_solution = solution;
      }
  }
} // namespace SolidDiffusion


int main()
{
  try
    {
      using namespace dealii;
      using namespace SolidDiffusion;

      DiffusionEquation<1> diffusion_equation_solver;
      diffusion_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
