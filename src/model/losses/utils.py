import torch
from scipy.optimize import linear_sum_assignment

from src.utils.plane import SymPlane


def calculate_cost_matrix_normals(points, y_pred, y_true):
    """
    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3])
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3])

    #return 1 - torch.abs(normals_true @ normals_pred.T)		# wrong!
    return 1 - torch.abs(normals_pred @ normals_true.T)


def calculate_cost_matrix_sde(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    cost_matrix = torch.zeros(y_pred.shape[0], y_true.shape[0])

    for i in range(y_pred.shape[0]):
        for j in range(y_true.shape[0]):
            pred_plane = SymPlane.from_tensor(y_pred[i, 0:6])
            true_plane = SymPlane.from_tensor(y_true[j, 0:6])
            cost_matrix[i, j] = torch.abs(
                torch.norm(
                    true_plane.reflect_points(points) - pred_plane.reflect_points(points),
                    dim=0
                )
            ).sum()
            if cost_matrix[i, j].isnan().any():
                print("Got cost matrix nan with planes:")
                print("pred", pred_plane)
                print("true", true_plane)

    return cost_matrix


def create_onehot(row_idx, length, device="cpu"):
    """

    :param device:
    :param row_idx: Array of index of matches
    :param length: length of the vector
    :return:
    """
    out = torch.zeros(length, device=device)
    out[row_idx] = 1
    return out


def get_optimal_assignment(points, y_pred, y_true, method):
    """

    :param method:
    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """
    m = y_pred.shape[0]
    cost_matrix = method(points, y_pred.detach().clone(), y_true)
    row_id, col_id = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    c_hat = create_onehot(row_id, m, device=points.device)
    y_pred = y_pred[row_id, :]
    return c_hat, y_pred


def calculate_angle_loss(y_pred, y_true):
    """
    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3], dim=1)  # M x 3
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3], dim=1)  # M x 3

    # cos(theta) = n_1 . n_2.T
    # => if n_1 == n_2 => cos(theta) = 1
    # or if n_1 == -n_2 => cos(Theta) = -1
    # Min theta <=> Min 1 - |cos(Theta)| <=> 1 - |n_1 . n_2.T|
    cos_angle = 1 - torch.abs(normals_true @ normals_pred.T) # M x M
    # We take the min of cos_angle because it is expected that the min value is
    # the one corresponding to the assigned plane, this is not a 100% certain but
    # it is expected because this plane was assigned to that true plane because of a reason
    # anyways this is fast but unreadable as #!=$@ idk how i would put it better tho
    return cos_angle.min(dim=0).values.mean()


def calculate_distance_loss(y_pred, y_true):
    """

    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    points_pred = y_pred[:, 3:6]
    points_true = y_true[:, 3:6]

    distances = torch.norm(points_true - points_pred, p=1, dim=0)

    return distances.mean()
