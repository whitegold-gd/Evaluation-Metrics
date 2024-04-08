import trimesh
import numpy as np
import time
import os
import visualisation
import open3d as o3d


def load_meshes_from_obj(file_paths, target_triangle_count):
    meshes = []
    for file_path in file_paths:
        print(f"Load object from {file_path}")
        mesh = trimesh.load_mesh(file_path)

        # Уменьшение количества треугольников
        mesh_o3d = o3d.io.read_triangle_mesh(file_path)
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_triangle_count)
        mesh.vertices = np.array(mesh_o3d.vertices)
        mesh.faces = np.array(mesh_o3d.triangles)

        meshes.append(mesh)
    return meshes


def chamfer_distance(mesh_ref, mesh_gen):
    _, distance1, _ = trimesh.proximity.closest_point(mesh_gen, mesh_ref.vertices)
    closest2, distance2, _ = trimesh.proximity.closest_point(mesh_ref, mesh_gen.vertices)

    tuple_list = [tuple(sublist) for sublist in closest2]
    unique_tuples = set(tuple_list)
    closest_vertices = [list(item) for item in unique_tuples]

    mean_distance1 = np.mean(np.abs(distance1))
    mean_distance2 = np.mean(np.abs(distance2))

    return mean_distance1 + mean_distance2, closest_vertices


def chamfer_coverage_mmd(meshes_ref, meshes_gen):
    total_distance = 0
    close_generated_vertices_count = 0
    reference_vertices_count = 0
    mmd_sum = 0
    times = []

    for i, mesh_gen in enumerate(meshes_gen):
        min_distance = float('inf')
        for j, mesh_ref in enumerate(meshes_ref):
            print(f"Compare gen_object:{i} and ref_object:{j}")

            start_time = time.time()
            distance, closest_vertices = chamfer_distance(mesh_ref, mesh_gen)
            times.append(time.time() - start_time)
            print(f"Chamfer distance function time: {time.time() - start_time} sec")

            print(f"Min distance: {distance}")
            print(f"Closest point on triangles for each point: {len(closest_vertices)}")
            print(f"Vertices count in reference object: {len(mesh_ref.vertices)}, "
                  f"in generated: {len(mesh_gen.vertices)} \n")

            total_distance += distance
            min_distance = min(min_distance, distance)
            reference_vertices_count += len(mesh_ref.vertices)
        if min_distance != float('inf'):
            close_generated_vertices_count += len(closest_vertices)
        mmd_sum += min_distance
    print(f'Mean time comparsion: {np.mean(times)}')

    chamfer = total_distance / (len(meshes_gen) * len(meshes_ref))
    coverage_share = close_generated_vertices_count / reference_vertices_count
    mmd = mmd_sum / reference_vertices_count
    return chamfer, coverage_share, mmd


def calculate_metrics(ref_meshes, gen_meshes):
    metrics = []
    cd, coverage, mmd = chamfer_coverage_mmd(ref_meshes, gen_meshes)
    metrics.append((cd, coverage, mmd))
    return metrics


def process_objects(ref_dir, gen_dir, target_triangle_count):
    ref_names = os.listdir(ref_dir)
    ref_paths = [os.path.join(ref_dir, name) for name in ref_names]
    ref_meshes = load_meshes_from_obj(ref_paths, target_triangle_count)

    gen_names = os.listdir(gen_dir)
    trunk_values = sorted(set(name.split("_")[0] for name in gen_names))

    metric_results = {trunk: [] for trunk in trunk_values}

    for trunk in trunk_values:
        print(f"Computing for trunk={trunk}")
        gen_paths = [os.path.join(gen_dir, name) for name in gen_names if name.startswith(trunk)]
        gen_meshes = load_meshes_from_obj(gen_paths, target_triangle_count)
        metrics = calculate_metrics(ref_meshes, gen_meshes)
        metric_results[trunk] = metrics

    return metric_results


if __name__ == "__main__":
    # pi_gan
    pi_gan_ref_dir = "pi_gan/ref_objects/"
    pi_gan_gen_dir = "pi_gan/gen_objects/"

    # eg3d
    eg3d_ref_dir = "eg3d/ref_objects/"
    eg3d_gen_dir = "eg3d/gen_objects/"

    target_triangle_count = 10000  # double to true count

    results_piGAN = process_objects(pi_gan_ref_dir, pi_gan_gen_dir, target_triangle_count)
    results_EG3D = process_objects(eg3d_ref_dir, eg3d_gen_dir, target_triangle_count)

    visualisation.plot_metrics(results_piGAN)

