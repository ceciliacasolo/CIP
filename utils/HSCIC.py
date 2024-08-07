import torch

class HSCIC:
    
    """ Hilbert-Schmidt Conditional Independence Criterion (HSCIC) implementation. """
    
    def __init__(self, regularization=0.1):

        self.regularization = regularization

    def __call__(self, Y, A, X):
        """
        Calculates the HSCIC loss for given tensors.
        
        Args:
            Y (torch.Tensor): Output tensor.
            A (torch.Tensor): Mediator variable tensor.
            X (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: The calculated HSCIC loss.
        """
        # Reshape and type cast tensors
        A, X, Y = [torch.FloatTensor(torch.reshape(t, [torch.Tensor.size(t)[0], -1])) for t in (A, X, Y)]

        # Calculate Gram matrices using Gaussian kernels
        gram_A, gram_X, gram_Y = [self.gaussian_kernel(t, t) for t in (A, X, Y)]

        # Compute HSCIC loss
        res_total = 0
        for i, row in enumerate(gram_X):
            res_i = self.inner_loss(torch.t(row), gram_A, gram_X, gram_Y)
            res_total += res_i

        return res_total / (i + 1)

    def inner_loss(self, X, gram_A, gram_X, gram_Y):
        """
        Helper function to compute inner loss component for HSCIC.
        
        Args:
            X (torch.Tensor): Transposed row of Gram matrix for X.
            gram_A (torch.Tensor): Gram matrix for A.
            gram_X (torch.Tensor): Gram matrix for X.
            gram_Y (torch.Tensor): Gram matrix for Y.
        
        Returns:
            torch.Tensor: Computed inner loss value.
        """
        n_samples = torch.Tensor.size(gram_Y)[0]
        identity = torch.eye(n_samples)
        W = gram_X + n_samples * self.regularization * identity

        # Solve linear system
        f = torch.linalg.solve(torch.t(W), X).reshape(-1, 1)
        fT = torch.t(f)

        # Calculate response
        res = torch.einsum('ij,jk,kl', fT, gram_A * gram_Y, f)
        res -= 2 * torch.einsum('ij,jk', fT, torch.einsum('ij,jk', gram_A, f) * torch.einsum('ij,jk', gram_Y, f))
        res += torch.einsum('ij,jk,kl', fT, gram_A, f) * torch.einsum('ij,jk,kl', fT, gram_Y, f)

        return res.flatten()

    def gaussian_kernel(self, a, b):
        """
        Computes a Gaussian kernel between tensors a and b.
        
        Args:
            a (torch.Tensor): Input tensor a.
            b (torch.Tensor): Input tensor b.
        
        Returns:
            torch.Tensor: The resulting Gaussian kernel matrix.
        """
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a, b = a.view(dim1_1, 1, depth), b.view(1, dim1_2, depth)
        squared_diff = (a - b).pow(2).mean(2) / (2 * 0.1 ** 2)
        return torch.exp(-squared_diff)
