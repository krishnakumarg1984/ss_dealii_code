#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

namespace Step52 {
using namespace dealii;

class Diffusion {
public:
  Diffusion();

  void run();

private:
  void setup_system();

  void assemble_system();

  double get_source(const double time, const Point<2> &point) const;

  Vector<double> evaluate_diffusion(const double time,
                                    const Vector<double> &y) const;

  void output_results(const double time, const unsigned int time_step,
                      TimeStepping::runge_kutta_method method) const;

  unsigned int
  embedded_explicit_method(const TimeStepping::runge_kutta_method method,
                           const unsigned int n_time_steps,
                           const double initial_time, const double final_time);

  const unsigned int fe_degree;

  const double diffusion_coefficient;
  const double absorption_cross_section;

  Triangulation<2> triangulation;

  const FE_Q<2> fe;

  DoFHandler<2> dof_handler;

  AffineConstraints<double> constraint_matrix;

  SparsityPattern sparsity_pattern;

  SparseMatrix<double> system_matrix;
  SparseMatrix<double> mass_matrix;

  SparseDirectUMFPACK inverse_mass_matrix;

  Vector<double> solution;
};

Diffusion::Diffusion()
    : fe_degree(3), diffusion_coefficient(1. / 30.),
      absorption_cross_section(1.), fe(fe_degree), dof_handler(triangulation) {}

void Diffusion::setup_system() {
  dof_handler.distribute_dofs(fe);

  VectorTools::interpolate_boundary_values(
      dof_handler, 1, Functions::ZeroFunction<2>(), constraint_matrix);
  constraint_matrix.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraint_matrix);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
}

void Diffusion::assemble_system() {
  system_matrix = 0.;
  mass_matrix = 0.;

  const QGauss<2> quadrature_formula(fe_degree + 1);

  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0.;
    cell_mass_matrix = 0.;

    fe_values.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_matrix(i, j) +=
              ((-diffusion_coefficient *                  // (-D
                    fe_values.shape_grad(i, q_point) *    //  * ∇ phi_i
                    fe_values.shape_grad(j, q_point)      //  * ∇ phi_j
                - absorption_cross_section *              //  -Sigma
                      fe_values.shape_value(i, q_point) * //  * phi_i
                      fe_values.shape_value(j, q_point))  //  * phi_j)
               * fe_values.JxW(q_point));                 // * dx
          cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(j, q_point) *
                                    fe_values.JxW(q_point);
        }

    cell->get_dof_indices(local_dof_indices);

    constraint_matrix.distribute_local_to_global(cell_matrix, local_dof_indices,
                                                 system_matrix);
    constraint_matrix.distribute_local_to_global(
        cell_mass_matrix, local_dof_indices, mass_matrix);
  }

  inverse_mass_matrix.initialize(mass_matrix);
}

double Diffusion::get_source(const double time, const Point<2> &point) const {
  const double intensity = 10.;
  const double frequency = numbers::PI / 10.;
  const double b = 5.;
  const double x = point(0);

  return intensity * (frequency * std::cos(frequency * time) * (b * x - x * x) +
                      std::sin(frequency * time) *
                          (absorption_cross_section * (b * x - x * x) +
                           2. * diffusion_coefficient));
}

Vector<double> Diffusion::evaluate_diffusion(const double time,
                                             const Vector<double> &y) const {
  Vector<double> tmp(dof_handler.n_dofs());
  tmp = 0.;
  system_matrix.vmult(tmp, y);

  const QGauss<2> quadrature_formula(fe_degree + 1);

  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  Vector<double> cell_source(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_source = 0.;

    fe_values.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double source =
          get_source(time, fe_values.quadrature_point(q_point));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_source(i) += fe_values.shape_value(i, q_point) * // phi_i(x)
                          source *                            // * S(x,t)
                          fe_values.JxW(q_point);             // * dx
    }

    cell->get_dof_indices(local_dof_indices);

    constraint_matrix.distribute_local_to_global(cell_source, local_dof_indices,
                                                 tmp);
  }

  Vector<double> value(dof_handler.n_dofs());
  inverse_mass_matrix.vmult(value, tmp);

  return value;
}

void Diffusion::output_results(const double time, const unsigned int time_step,
                               TimeStepping::runge_kutta_method method) const {
  std::string method_name;

  switch (method) {
  case TimeStepping::DOPRI: {
    method_name = "dopri";
    break;
  }
  default: {
    std::cout << "This time-stepping method is not implemented. Exiting ..."
              << std::endl;
    break;
  }
  }

  DataOut<2> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  data_out.set_flags(DataOutBase::VtkFlags(time, time_step));

  const std::string filename = "solution_" + method_name + "-" +
                               Utilities::int_to_string(time_step, 3) + ".vtu";
  std::ofstream output(filename);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;

  static std::string method_name_prev = "";
  static std::string pvd_filename;
  if (method_name_prev != method_name) {
    times_and_names.clear();
    method_name_prev = method_name;
    pvd_filename = "solution_" + method_name + ".pvd";
  }
  times_and_names.emplace_back(time, filename);
  std::ofstream pvd_output(pvd_filename);
  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

unsigned int Diffusion::embedded_explicit_method(
    const TimeStepping::runge_kutta_method method,
    const unsigned int n_time_steps, const double initial_time,
    const double final_time) {
  double time_step =
      (final_time - initial_time) / static_cast<double>(n_time_steps);
  double time = initial_time;
  const double coarsen_param = 1.2;
  const double refine_param = 0.8;
  const double min_delta = 1e-8;
  const double max_delta = 10 * time_step;
  const double refine_tol = 1e-1;
  const double coarsen_tol = 1e-5;

  solution = 0.;
  constraint_matrix.distribute(solution);

  TimeStepping::EmbeddedExplicitRungeKutta<Vector<double>>
      embedded_explicit_runge_kutta(method, coarsen_param, refine_param,
                                    min_delta, max_delta, refine_tol,
                                    coarsen_tol);
  output_results(time, 0, method);

  unsigned int n_steps = 0;
  while (time < final_time) {
    if (time + time_step > final_time)
      time_step = final_time - time;

    time = embedded_explicit_runge_kutta.evolve_one_time_step(
        [this](const double time, const Vector<double> &y) {
          return this->evaluate_diffusion(time, y);
        },
        time, time_step, solution);

    constraint_matrix.distribute(solution);

    if ((n_steps + 1) % 10 == 0)
      output_results(time, n_steps + 1, method);

    time_step = embedded_explicit_runge_kutta.get_status().delta_t_guess;
    ++n_steps;
  }

  return n_steps;
}

void Diffusion::run() {
  GridGenerator::hyper_cube(triangulation, 0., 5.);
  triangulation.refine_global(4);

  for (const auto &cell : triangulation.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary()) {
        if ((face->center()[0] == 0.) || (face->center()[0] == 5.))
          face->set_boundary_id(1);
        else
          face->set_boundary_id(0);
      }

  setup_system();

  assemble_system();

  unsigned int n_steps = 0;
  const unsigned int n_time_steps = 200;
  const double initial_time = 0.;
  const double final_time = 10.;

  n_steps = embedded_explicit_method(TimeStepping::DOPRI, n_time_steps,
                                     initial_time, final_time);
  std::cout << "   Dopri:                    error=" << solution.l2_norm()
            << std::endl;
  std::cout << "                steps performed=" << n_steps << std::endl;
}
} // namespace Step52

int main() {
  try {
    Step52::Diffusion diffusion;
    diffusion.run();
  } catch (std::exception &exc) {
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
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  };

  return 0;
}
