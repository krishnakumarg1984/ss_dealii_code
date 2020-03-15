template <int dim>
void DiffusionEquation<dim>::assemble_neumann_rhs()
{
  QGauss<dim - 1> face_quadrature_formula(fe->degree + 1);

  const unsigned int n_face_q_points = face_quadrature_formula.size();

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FEFaceValues<dim> fe_face_values(*fe,
                                   face_quadrature_formula,
                                   update_values | update_JxW_values);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0.;
      fe_values.reinit(cell);
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && (face->boundary_id() == 1))
          {
            fe_face_values.reinit(cell, face);
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
              {
                double neumann_value = -0.1;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) +=
                    (neumann_value *                          // g(x_q)
                     fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                     fe_face_values.JxW(q_point));            // dx
              }
          }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
}

// DiffusionCoefficient<dim> rhs_function;
// rhs_function.set_time(time);
// VectorTools::create_right_hand_side(dof_handler,
//                                     QGauss<dim>(fe.degree + 1),
//                                     // rhs_function,
//                                     tmp);
// forcing_terms = tmp;
// forcing_terms *= time_step * theta;
// rhs_function.set_time(time - time_step);
// VectorTools::create_right_hand_side(dof_handler,
//                                     QGauss<dim>(fe.degree + 1),
//                                     // rhs_function,
//                                     tmp);
// forcing_terms.add(time_step * (1 - theta), tmp);
// system_rhs += forcing_terms;
