/* ---------------------------------------------------------------------
 * Copyright (C) 2020 by Krishnakumar Gopalakrishnan
 * Authors: Krishnakumar Gopalakrishnan, University College London, 2020
 */


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

namespace SSBatteryScaledDiffusionEqn
{
  using namespace dealii;


  template <int dim>
  class SolidDiffusion
  {
  public:
    SolidDiffusion();

    void run();

  private:
    void setup_system();

    void assemble_system(); // Assembles all components of matrix i.e. those
    // multiplying U(t)

    // This gives the numerical value of the BC (at left boundary in 1D case)
    double get_boundary_neumann_value(const double time) const;

    // In the implementation code of the below function, the get_source()
    // function below needs to be evaluated only if the MMS_flag is true
    double get_source(const double time, const Point<dim> &point) const;


    // The 'evaluate_diffusion' function below evaluates RHS of weak form of
    // spatially discretised PDE (M^-1  (-Dy-Ay+S + 𝛽(t) 𝜓(0)) ). Do not forget
    // that this function's return value already accounts for M^-1.

    // In the implementation of the below function, the section/logic for S(t)
    // from get_source() needs evaluation only if the MMS_flag is true
    Vector<double> evaluate_diffusion(const double          time,
                                      const Vector<double> &y) const;

    void output_results(const double                     time,
                        const unsigned int               time_step,
                        TimeStepping::runge_kutta_method method) const;

    unsigned int
    embedded_explicit_method(const TimeStepping::runge_kutta_method method,
                             const unsigned int n_time_steps,
                             const double       initial_time,
                             const double       final_time);


    const unsigned int fe_degree;

    const double diffusion_coefficient;
    const double absorption_cross_section;

    Triangulation<dim> triangulation;

    const FE_Q<dim> fe;

    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraint_matrix;

    SparsityPattern sparsity_pattern;

    SparseMatrix<double> system_matrix; // - \mathcal{D} - \mathcal{A}
    SparseMatrix<double> mass_matrix;

    SparseDirectUMFPACK inverse_mass_matrix;

    Vector<double> solution;

    const bool   mms_flag;
    const double b; // length of the hypercube domain in each dimension
  };


  template <int dim>
  SolidDiffusion<dim>::SolidDiffusion()
    : fe_degree(3)
    , diffusion_coefficient(1. / 25.) // value of D
    // , diffusion_coefficient(1. / 30.) // value of D
    , absorption_cross_section(0.) // value of \Sigma_a
    , fe(fe_degree)
    , dof_handler(triangulation)
    , mms_flag(false)
    // , mms_flag(true) // if true, will run MMS method with the preassumed
    // analytical soln (& hence, preassumed analytical source term) in the eqn
    , b(5.0)
  {}


  template <int dim>
  void SolidDiffusion<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    constraint_matrix.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraint_matrix);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void SolidDiffusion<dim>::assemble_system() // Assembles all matrix components
  // i.e. those multiplying U(t)
  {
    system_matrix = 0.;
    mass_matrix   = 0.;

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix      = 0.;
        cell_mass_matrix = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                cell_matrix(i, j) +=
                  ((-diffusion_coefficient *                // (-D
                      fe_values.shape_grad(i, q_point) *    //  * ∇ phi_i
                      fe_values.shape_grad(j, q_point)      //  * ∇ phi_j
                    - absorption_cross_section *            //  -Sigma
                        fe_values.shape_value(i, q_point) * //  * phi_i
                        fe_values.shape_value(j, q_point))  //  * phi_j)
                   * fe_values.JxW(q_point));               // * dx
                cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                          fe_values.shape_value(j, q_point) *
                                          fe_values.JxW(q_point);
              }

        cell->get_dof_indices(local_dof_indices);

        constraint_matrix.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
        constraint_matrix.distribute_local_to_global(cell_mass_matrix,
                                                     local_dof_indices,
                                                     mass_matrix);
      }

    inverse_mass_matrix.initialize(mass_matrix);
  }

  template <int dim>
  double
  SolidDiffusion<dim>::get_boundary_neumann_value(const double time) const
  {
    return 0.5; // for now, hard-coding some neumannBC value
    // return 0.0; // for now, hard-coding zero neumannBC value (for MMS
    // checking etc.)
  }

  // This template function shall be called (from within evaluate_diffusion()),
  // but only if the MMS_flag is active
  template <int dim>
  double SolidDiffusion<dim>::get_source(const double      time,
                                         const Point<dim> &point) const
  {
    const double intensity = 10.;               // A (amplitude)
    const double frequency = numbers::PI / 10.; // omega
    const double b         = 5.; // length in each co-ordinate direction (repeat
    // declaration in GridGenerator::hyper_cube method)
    const double x = point(0); // assign the x-coord of point to 'x'

    // chosen analytical solution:u(x,t) is :
    // A * sin (omega*t) * exp(-((x-0.5*b).^2)/(0.125*b))

    // So the S(x,t) will be the following:
    return intensity * exp(-2.0 * pow(b - 2.0 * x, 2) / b) *
           (frequency * std::cos(frequency * time) +
            (absorption_cross_section -
             (16.0 * diffusion_coefficient *
              (((4.0 * pow(b - 2.0 * x, 2)) / b) - 1.0) / b)) *
              std::sin(frequency * time));
  }

  // This template function shall be called only if the MMS_flag is INACTIVE
  // Currently set up for only for 1D case. This is highly problem-specific
  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time)
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
      const double x         = p(0);
      const double b         = 5.0;
      const double amplitude = 1.0;

      // try various manually chosen spatial function(s) for the IC

      // Gaussian curve centred at half the domain
      // return amplitude * exp(-(8.0 * pow(x - 0.5 * b, 2) / b));

      // Gaussian curve centred at 1/3rd of the domain length
      // return amplitude * exp(-(2.0 * pow(x - 0.3 * b, 2) / b));

      // Spatially constant IC
      return amplitude;
    }
  };



  // Ensure that the source S(t) is evaluated only when the MMS_flag is active
  template <int dim>
  Vector<double>
  SolidDiffusion<dim>::evaluate_diffusion(const double          time,
                                          const Vector<double> &y) const
  {
    Vector<double> tmp(dof_handler.n_dofs());
    tmp = 0.;
    system_matrix.vmult(tmp, y); // tmp now has (-mathcal{D} - \mathcal{A}) * y

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_JxW_values);

    const unsigned int n_face_q_points = face_quadrature_formula.size();

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Vector<double> cell_neumannbc_contribution(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_neumannbc_contribution = 0.;
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
                  const double left_boundary_neumann_value =
                    get_boundary_neumann_value(time);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_neumannbc_contribution(i) +=
                      (left_boundary_neumann_value *            // \beta(t)
                       fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                       fe_face_values.JxW(q_point));            // dx
                }
            }

        cell->get_dof_indices(local_dof_indices);

        // tmp+=NeumannBC; i.e. (-D - A)y + NeumannBC
        constraint_matrix.distribute_local_to_global(
          cell_neumannbc_contribution, local_dof_indices, tmp);

        // Evaluate only if MMS_flag is active (non-zero pre-computed S(t))
        if (mms_flag == true)
          {
            Vector<double> cell_source(dofs_per_cell);

            cell_source = 0.;
            fe_values.reinit(cell);
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                const double source =
                  get_source(time, fe_values.quadrature_point(q_point));
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_source(i) +=
                    fe_values.shape_value(i, q_point) * // phi_i(x)
                    source *                            // * S(x,t)
                    fe_values.JxW(q_point);             // * dx
              }

            // tmp+=S(t); i.e. (-D - A)y + NeumannBC + S(t)
            constraint_matrix.distribute_local_to_global(cell_source,
                                                         local_dof_indices,
                                                         tmp);
          }
      }


    Vector<double> value(dof_handler.n_dofs());
    inverse_mass_matrix.vmult(value, tmp);

    return value; // value contains -M^-1 * (-Dy - Ay + NeumannBC + optionally,
                  // S)
  }


  template <int dim>
  void SolidDiffusion<dim>::output_results(
    const double                     time,
    const unsigned int               time_step,
    TimeStepping::runge_kutta_method method) const
  {
    std::string method_name;

    switch (method)
      {
          case TimeStepping::DOPRI: {
            method_name = "dopri";
            break;
          }
          default: {
            std::cout
              << "This time-stepping method is not implemented. Exiting ..."
              << std::endl;
            break;
          }
      }

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, time_step));

    const std::string filename = "solution_" + method_name + "-" +
                                 Utilities::int_to_string(time_step, 3) +
                                 ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;

    static std::string method_name_prev = "";
    static std::string pvd_filename;
    if (method_name_prev != method_name)
      {
        times_and_names.clear();
        method_name_prev = method_name;
        pvd_filename     = "solution_" + method_name + ".pvd";
      }
    times_and_names.emplace_back(time, filename);
    std::ofstream pvd_output(pvd_filename);
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }


  template <int dim>
  unsigned int SolidDiffusion<dim>::embedded_explicit_method(
    const TimeStepping::runge_kutta_method method,
    const unsigned int                     n_time_steps,
    const double                           initial_time,
    const double                           final_time)
  {
    double time_step =
      (final_time - initial_time) / static_cast<double>(n_time_steps);
    double       time          = initial_time;
    const double coarsen_param = 1.2;
    const double refine_param  = 0.8;
    const double min_delta     = 1e-8;
    const double max_delta     = 10 * time_step;
    const double refine_tol    = 1e-1;
    const double coarsen_tol   = 1e-5;

    if (mms_flag)
      solution = 0.;

    constraint_matrix.distribute(solution);

    TimeStepping::EmbeddedExplicitRungeKutta<Vector<double>>
      embedded_explicit_runge_kutta(method,
                                    coarsen_param,
                                    refine_param,
                                    min_delta,
                                    max_delta,
                                    refine_tol,
                                    coarsen_tol);
    output_results(time, 0, method);

    unsigned int n_steps = 0;
    while (time < final_time)
      {
        if (time + time_step > final_time)
          time_step = final_time - time;

        time = embedded_explicit_runge_kutta.evolve_one_time_step(
          [this](const double time, const Vector<double> &y) {
            return this->evaluate_diffusion(time, y);
          },
          time,
          time_step,
          solution);

        constraint_matrix.distribute(solution);

        if ((n_steps + 1) % 10 == 0)
          output_results(time, n_steps + 1, method);

        time_step = embedded_explicit_runge_kutta.get_status().delta_t_guess;
        ++n_steps;
      }

    return n_steps;
  }


  template <int dim>
  void SolidDiffusion<dim>::run()
  {
    GridGenerator::hyper_cube(triangulation, 0., b); // b = 5 for now
    triangulation.refine_global(6);

    setup_system();
    if (mms_flag == false)
      {
        VectorTools::project(dof_handler,
                             constraint_matrix,
                             QGauss<dim>(fe.degree + 1),
                             // InitialValues<1>(1, time),
                             InitialValues<dim>(1, 0),
                             solution);
      }

    assemble_system(); // Assembles matrix components (those multiplying U(t))

    unsigned int       n_steps      = 0;
    const unsigned int n_time_steps = 200;
    const double       initial_time = 0.;
    const double       final_time   = 10.;
    // In the MMS version of the code, for the final_time above, note that the
    // (frequency) omega = pi/10; sin(omega*t_final) = sin(pi) = 0

    n_steps = embedded_explicit_method(TimeStepping::DOPRI,
                                       n_time_steps,
                                       initial_time,
                                       final_time);

    // In this context, the error is valid only if the MMS_flag is active
    if (mms_flag)
      {
        std::cout << "   Dopri:                    error=" << solution.l2_norm()
                  << std::endl;
        std::cout << "                steps performed=" << n_steps << std::endl;
      }
  }
} // namespace SSBatteryScaledDiffusionEqn



int main()
{
  try
    {
      using namespace dealii;
      using namespace SSBatteryScaledDiffusionEqn;
      SolidDiffusion<1> soliddiffusion;
      soliddiffusion.run();
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
    };

  return 0;
}
