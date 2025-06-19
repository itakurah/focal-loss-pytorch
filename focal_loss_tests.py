import unittest
import torch

from focal_loss import FocalLoss


class TestFocalLoss(unittest.TestCase):

    def test_binary_focal_loss(self):
        """ Test the FocalLoss with binary classification. """
        criterion = FocalLoss(gamma=2, alpha=0.25, task_type='binary')
        inputs = torch.randn(16)  # Logits from the model (batch_size=16)
        targets = torch.randint(0, 2, (16,)).float()  # Binary ground truth (0 or 1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss(self):
        """ Test the FocalLoss with multi-class classification. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=[0.25] * num_classes, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, num_classes, (16,))  # Ground truth with integer class labels (0 to num_classes-1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss_soft_labels(self):
        """ Test the FocalLoss with multi-class classification and soft labels. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=[0.25] * num_classes, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)
        # Soft labels: each row sums to 1
        targets = torch.softmax(torch.randn(16, num_classes), dim=1)
        loss = criterion.multi_class_focal_loss(inputs, targets)
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)

    def test_multi_label_focal_loss(self):
        """ Test the FocalLoss with multi-label classification. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-label')
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, 2, (16, num_classes)).float()  # Multi-label ground truth (0 or 1 for each class)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_binary_focal_loss_no_alpha(self):
        """ Test the FocalLoss with binary classification without alpha. """
        criterion = FocalLoss(gamma=2, task_type='binary')
        inputs = torch.randn(16)  # Logits from the model (batch_size=16)
        targets = torch.randint(0, 2, (16,)).float()  # Binary ground truth (0 or 1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss_no_alpha(self):
        """ Test the FocalLoss with multi-class classification without alpha. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, num_classes, (16,))  # Ground truth with integer class labels (0 to num_classes-1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_label_focal_loss_no_alpha(self):
        """ Test the FocalLoss with multi-label classification without alpha. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, task_type='multi-label')
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, 2, (16, num_classes)).float()  # Multi-label ground truth (0 or 1 for each class)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_invalid_task_type(self):
        """ Test FocalLoss with an invalid task type """
        with self.assertRaises(ValueError):
            criterion = FocalLoss(gamma=2, task_type='invalid-task')
            inputs = torch.randn(16, 5)
            targets = torch.randint(0, 5, (16,))
            criterion(inputs, targets)

    def test_reduction_modes(self):
        """Test sum and none reduction modes for binary, multi-class, and multi-label."""
        inputs = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        for reduction in ['sum', 'none']:
            criterion = FocalLoss(gamma=2, task_type='binary', reduction=reduction)
            loss = criterion(inputs, targets)
            if reduction == 'sum':
                self.assertEqual(loss.dim(), 0)  # scalar
            else:
                self.assertEqual(loss.shape, (8,))  # per-sample

        num_classes = 3
        inputs_mc = torch.randn(8, num_classes)
        targets_mc = torch.randint(0, num_classes, (8,))
        for reduction in ['sum', 'none']:
            criterion = FocalLoss(gamma=2, task_type='multi-class', num_classes=num_classes, reduction=reduction)
            loss = criterion(inputs_mc, targets_mc)
            if reduction == 'sum':
                self.assertEqual(loss.dim(), 0)
            else:
                self.assertEqual(loss.shape, (8,))  # per-sample

        inputs_ml = torch.randn(8, num_classes)
        targets_ml = torch.randint(0, 2, (8, num_classes)).float()
        for reduction in ['sum', 'none']:
            criterion = FocalLoss(gamma=2, task_type='multi-label', reduction=reduction)
            loss = criterion(inputs_ml, targets_ml)
            if reduction == 'sum':
                self.assertEqual(loss.dim(), 0)
            else:
                self.assertEqual(loss.shape, (8, num_classes))  # per-sample per-label

    def test_multi_label_soft_targets(self):
        """Test multi-label focal loss with soft probabilistic targets."""
        num_classes = 4
        criterion = FocalLoss(gamma=2, task_type='multi-label')
        inputs = torch.randn(8, num_classes)
        targets = torch.rand(8, num_classes)  # continuous in [0,1]
        loss = criterion(inputs, targets)
        self.assertTrue(loss.item() >= 0)

    def test_alpha_tensor_for_multi_label(self):
        """Test multi-label with per-class alpha tensor."""
        num_classes = 4
        alpha = torch.linspace(0.1, 1.0, num_classes)
        criterion = FocalLoss(gamma=2, alpha=alpha, task_type='multi-label')
        inputs = torch.randn(8, num_classes)
        targets = torch.randint(0, 2, (8, num_classes)).float()
        loss = criterion(inputs, targets)
        self.assertTrue(loss.item() >= 0)

    def test_alpha_scalar_for_multi_class(self):
        """Test multi-class with a scalar alpha value for all classes."""
        num_classes = 3
        criterion = FocalLoss(gamma=2, alpha=0.5, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(8, num_classes)
        targets = torch.randint(0, num_classes, (8,))
        loss = criterion(inputs, targets)
        self.assertTrue(loss.item() >= 0)

    def test_missing_num_classes(self):
        """Test that num_classes is required for multi-class if alpha is a list."""
        with self.assertRaises(AssertionError):
            FocalLoss(gamma=2, alpha=[0.1, 0.2], task_type='multi-class')  # missing num_classes

    def test_extreme_logits_and_gamma_zero(self):
        """Test with extreme logits and gamma=0 behavior (standard CE)."""
        num_classes = 3
        inputs = torch.full((8, num_classes), 1000.0)  # extreme large logit
        targets = torch.randint(0, num_classes, (8,))
        criterion = FocalLoss(gamma=0, task_type='multi-class', num_classes=num_classes)
        loss = criterion(inputs, targets)
        self.assertTrue(loss.item() >= 0)

    def test_backward(self):
        """Test backward pass to ensure differentiability."""
        num_classes = 3
        inputs = torch.randn(8, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (8,))
        criterion = FocalLoss(gamma=2, task_type='multi-class', num_classes=num_classes)
        loss = criterion(inputs, targets)
        loss.backward()
        self.assertIsNotNone(inputs.grad)



if __name__ == '__main__':
    unittest.main()
