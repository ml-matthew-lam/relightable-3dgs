# Relightable 3D Gaussian Splatting

$$q = (q_w, q_x, q_y, q_z)$$

$$
R = \begin{bmatrix}
    1-2(q_y^2+q_z^2) & 2(q_xq_y-q_wq_z) & 2(q_xq_z+q_wq_y)\\
    2(q_xq_y+q_wq_z) & 1-2(q_x^2+q_z^2) & 2(q_yq_z-q_wq_x)\\
    2(q_xq_z-q_wq_y) & 2(q_yq_z+q_wq_x) & 1-2(q_x^2+q_y^2)
    \end{bmatrix}
$$

$$ C = \rho \max(0, \mathbf n \cdot \mathbf l) $$

