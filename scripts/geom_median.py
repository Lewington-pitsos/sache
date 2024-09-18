import boto3
import json
import torch

from types import SimpleNamespace

import numpy as np
import tqdm

from sache.constants import MB, BUCKET_NAME
from sache.cache import S3RCache

def geometric_median_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:`n`, whose elements are each a ``torch.Tensor`` of shape ``(d,)``
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value
            norms = torch.stack([torch.linalg.norm((p - median).view(-1)) for p in points])
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)

            objective_value = geometric_median_objective(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break
            
            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )


def geometric_median_list_of_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``torch.Tensor``.
        Each inner list has the same "shape".
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``torch.Tensor`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        median = weighted_average(points, weights)
        new_weights = weights
        objective_value = geometric_median_objective(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        for _ in range(maxiter):
            prev_obj_value = objective_value
            denom = torch.stack([l2distance(p, median) for p in points])
            new_weights = weights / torch.clamp(denom, min=eps) 
            median = weighted_average(points, new_weights)

            objective_value = geometric_median_objective(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break
        
    median = weighted_average(points, new_weights)  # for autodiff

    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def weighted_average_component(points, weights):
    ret = points[0] * weights[0]
    for i in range(1, len(points)):
        ret += points[i] * weights[i]
    return ret

def weighted_average(points, weights):
    weights = weights / weights.sum()
    return [weighted_average_component(component, weights=weights) for component in zip(*points)]

@torch.no_grad()
def geometric_median_objective(median, points, weights):
    return np.average([l2distance(p, median).item() for p in points], weights=weights.cpu())

@torch.no_grad()
def l2distance(p1, p2):
    return torch.linalg.norm(torch.stack([torch.linalg.norm(x1 - x2) for (x1, x2) in zip(p1, p2)]))


def geometric_median_per_component(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``numpy.ndarray``.
        Each inner list has the same "shape".
    :param weights: ``numpy.ndarray`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``numpy.ndarray`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated, one for each component. 
        - `logs`: function values encountered through the course of the algorithm.
    """
    components = list(zip(*points))
    median = []
    termination = []
    logs = []
    new_weights = []
    pbar = tqdm.tqdm(components)
    for component in pbar:
        ret = geometric_median_array(component, weights, eps, maxiter, ftol)
        median.append(ret.median)
        new_weights.append(ret.new_weights)
        termination.append(ret.termination)
        logs.append(ret.logs)
    return SimpleNamespace(median=median, termination=termination, logs=logs)

def weighted_average(points, weights):
    weights = weights / weights.sum()
    ret = points[0] * weights[0]
    for i in range(1, len(points)):
        ret += points[i] * weights[i]
    return ret

@torch.no_grad()
def geometric_median_objective(median, points, weights):
    return np.average([torch.linalg.norm((p - median).reshape(-1)).item() for p in points], weights=weights.cpu())


def compute_geometric_median(
	points,
	eps=1e-6, maxiter=100, ftol=1e-20
):
	""" Compute the geometric median of points `points` with weights given by `weights`. 
	"""
	if type(points) == torch.Tensor:
		# `points` are given as an array of shape (n, d)
		points = [p for p in points]  # translate to list of arrays format
	if type(points[0]) == torch.Tensor: # `points` are given in list of arrays format
		weights = torch.ones(len(points), device=points[0].device)
		to_return = geometric_median_array(points, weights, eps, maxiter, ftol)
	elif type(points[0]) in [list, tuple]: # `points` are in list of list of arrays format
		weights = torch.ones(len(points), device=points[0][0].device)
		to_return = geometric_median_list_of_array(points, weights, eps, maxiter, ftol)
	else:
		raise ValueError(f"Unexpected format {type(points[0])} for list of list format.")
	return to_return
		
if __name__ == "__main__":
    with open('.credentials.json') as f:
        credentials = json.load(f)
    
    data_name = 'ViT-3_000_000'
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    cache = S3RCache(s3_client, data_name, BUCKET_NAME, chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=8)

    all_activations = None

    for batch in cache:
        if all_activations is None:
            all_activations = batch
        else:
            all_activations = torch.cat([all_activations, batch], dim=0)

        if all_activations.size(0) > 1_000_000:
            break

    out = compute_geometric_median(
            all_activations, maxiter=100).median

    torch.save(out, 'geom_median.pt')

    # previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
    # distances = torch.norm(all_activations - out, dim=-1)

    # print("Reinitializing b_dec with geometric median of activations")
    # print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
    # print(f"New distances: {distances.median(0).values.mean().item()}")

    # out = torch.tensor(out, dtype=self.dtype, device=self.device)
    # self.b_dec.data = out