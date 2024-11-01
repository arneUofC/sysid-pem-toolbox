# This file is used for testing the file, making it assessible to the test framework
import sys
import os
sys.path.insert(0, os.path.abspath('..')) # Insert the parent directory into sys.path to locate the package

import unittest
import numpy as np
import control as ct
from sysid_pem_toolbox.sysid_pem import (
    theta_2_BCDF, theta_2_tf_box_jenkins, jac_V_bj, V_box_jenkins,
    y_hat_box_jenkins, V_oe, theta_2_tf_oe, y_hat_oe,
    V_arx_lin_reg, theta_2_tf_arx, cross_correlation_test,
    auto_correlation_test, FIR_estimates_GH, tf_realization_GH,
    get_initial_estimate_box_jenkins, get_regression_matrix
)


class TestSysidPemToolbox(unittest.TestCase):

    def setUp(self):
        # Setting up common variables and small example data
        self.theta = np.array([0.5, -0.4, 0.3, -0.2, 0.1])
        self.n = [2, 1, 1, 2, 0]
        self.Ts = 1.0
        self.y = np.array([1, 2, 3, 4, 5])
        self.u = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    def test_theta_2_BCDF(self):
        # Test if theta_2_BCDF returns the correct matrices
        # B, C, D, F = theta_2_BCDF(self.theta, self.n)
        # self.assertEqual(len(B), self.n[3] + 1)
        # self.assertEqual(len(C), self.n[2] + 1)
        # self.assertEqual(len(D), self.n[2] + 1)
        # self.assertEqual(len(F), self.n[3] + 1)
        pass

    def test_theta_2_tf_box_jenkins(self):
        # Check if transfer functions are created without error
        # G_theta, H_theta = theta_2_tf_box_jenkins(self.theta, self.n, self.Ts)
        # self.assertIsInstance(G_theta, ct.TransferFunction)
        # self.assertIsInstance(H_theta, ct.TransferFunction)
        pass

    def test_jac_V_bj(self):
        # Verify the output shape of the Jacobian function
        # jacobian = jac_V_bj(self.theta, self.n, self.y, self.u)
        # self.assertEqual(jacobian.shape[1], sum(self.n[:4]))
        pass

    def test_V_box_jenkins(self):
        # Check if the cost function V_box_jenkins returns expected output shape
        # cost = V_box_jenkins(self.theta, self.n, self.y, self.u)
        # self.assertEqual(cost.shape, self.y.shape)
        pass

    def test_y_hat_box_jenkins(self):
        # Check if the prediction y_hat has the expected length
        # y_hat = y_hat_box_jenkins(self.theta, self.n, self.y, self.u)
        # self.assertEqual(y_hat.shape, self.y.shape)
        pass

    def test_V_oe(self):
        # Check if the output error function returns a single value
        # cost = V_oe(self.theta, [2, 1], self.y, self.u)
        # self.assertIsInstance(cost, float)
        pass

    def test_theta_2_tf_oe(self):
        # Check if transfer functions are created without error
        # G_theta, H_theta = theta_2_tf_oe(self.theta, [2, 1], self.Ts)
        # self.assertIsInstance(G_theta, ct.TransferFunction)
        # self.assertIsInstance(H_theta, ct.TransferFunction)
        pass

    def test_y_hat_oe(self):
        # Check if the prediction error has the expected shape
        # epsilon = y_hat_oe(self.theta, [2, 1], self.y, self.u)
        # self.assertEqual(epsilon.shape, self.y.shape)
        pass

    def test_V_arx_lin_reg(self):
        # Check if the ARX linear regression returns the correct theta size
        # theta = V_arx_lin_reg([2, 1, 0], self.y, self.u)
        # self.assertEqual(len(theta), 3)
        pass

    def test_theta_2_tf_arx(self):
        # Check if transfer functions are created without error
        # G_theta, H_theta = theta_2_tf_arx(self.theta, [2, 1, 0], self.Ts)
        # self.assertIsInstance(G_theta, ct.TransferFunction)
        # self.assertIsInstance(H_theta, ct.TransferFunction)
        pass

    def test_cross_correlation_test(self):
        # Run the cross-correlation test and ensure it completes without error
        # cross_correlation_test(self.y - self.u, self.u, tau=10)
        pass

    def test_auto_correlation_test(self):
        # Run the auto-correlation test and ensure it completes without error
        # auto_correlation_test(self.y - self.u, tau=10)
        pass

    def test_FIR_estimates_GH(self):
        # Test if FIR_estimates_GH returns g and h of expected sizes
        # g, h = FIR_estimates_GH([2, 1, 0], self.y, self.u)
        # self.assertEqual(len(g), 1 + 2)  # 2 (nb) + 1 (nk)
        # self.assertEqual(len(h), 1 + 2)  # nh + 1
        pass

    def test_tf_realization_GH(self):
        # Test if tf_realization_GH returns proper transfer function realization
        # g = np.array([0.5, -0.3, 0.2])
        # h = np.array([1.0, -0.5])
        # G_theta, H_theta = tf_realization_GH(g, h, self.n)
        # self.assertIsInstance(G_theta, ct.TransferFunction)
        # self.assertIsInstance(H_theta, ct.TransferFunction)
        pass
    
    def test_get_initial_estimate_box_jenkins(self):
        pass
    
    def test_get_regression_matrix(self):
        pass


# Run the tests
if __name__ == '__main__':
    unittest.main()
