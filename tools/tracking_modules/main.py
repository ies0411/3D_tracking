import os, numpy as np, time, sys, argparse
from utils import Config, get_subfolder_seq, initialize, createFolder
from io import (
    load_detection,
    get_saving_dir,
    get_frame_det,
    save_results,
    save_affinity,
)
from model import AB3DMOT


def main_per_cat(cfg, cat, ID_start):

    # get data-cat-split specific path
    # result_sha = "%s_%s_%s" % (cfg.det_name, cat, cfg.split)
    # print(result_sha)
    # det_root = os.path.join("./data", cfg.dataset, "detection", result_sha)
    # subfolder, det_id2str, hw, seq_eval, data_root = get_subfolder_seq(
    #     cfg.dataset, cfg.split
    # )
    # trk_root = os.path.join(data_root, "tracking")
    # save_dir = os.path.join(cfg.save_root, result_sha + "_H%d" % cfg.num_hypo)
    # createFolder(save_dir)

    # create eval dir for each hypothesis
    # eval_dir_dict = dict()
    # for index in range(cfg.num_hypo):
    #     eval_dir_dict[index] = os.path.join(save_dir, "data_%d" % index)
    #     createFolder(eval_dir_dict[index])

    # loop every sequence
    # seq_count = 0
    total_time = 0.0
    # for seq_name in seq_eval:
    # seq_file = os.path.join(det_root, seq_name + ".txt")
    # seq_dets, flag = load_detection(seq_file)  # load detection
    # TODO : change detection type : file -> data
    # if not flag:
    #     continue  # no detection

    # create folders for saving
    # eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = get_saving_dir(
    #     eval_dir_dict, seq_name, save_dir, cfg.num_hypo
    # )
    # initialize tracker
    # tracker, frame_list = initialize(cfg, cat, ID_start)
    tracker = AB3DMOT(cfg, cat, ID_init=ID_start)

    # loop over frame
    # min_frame, max_frame = int(frame_list[0]), int(frame_list[-1])

    # for frame in range(min_frame, max_frame + 1):
    # add an additional frame here to deal with the case that the last frame, although no detection
    # but should output an N x 0 affinity for consistency

    # tracking by detection
    dets_frame = get_frame_det(seq_dets)
    since = time.time()
    results, affi = tracker.track(dets_frame)
    total_time += time.time() - since

    # saving affinity matrix, between the past frame and current frame
    # e.g., for 000006.npy, it means affinity between frame 5 and 6
    # note that the saved value in affinity can be different in reality because it is between the
    # original detections and ego-motion compensated predicted tracklets, rather than between the
    # actual two sets of output tracklets
    # save_affi_file = os.path.join(affinity_dir, "%06d.npy" % frame)
    # save_affi_vis = os.path.join(affinity_vis, "%06d.txt" % frame)
    # if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0):
    #     # save affinity as long as there are tracklets in at least one frame
    #     np.save(save_affi_file, affi)

    #     # cannot save for visualization unless both two frames have tracklets
    #     if affi.shape[0] > 0 and affi.shape[1] > 0:
    #         save_affinity(affi, save_affi_vis)

    # saving trajectories, loop over each hypothesis
    # for hypo in range(cfg.num_hypo):
    #     save_trk_file = os.path.join(save_trk_dir[hypo], "%06d.txt" % frame)
    #     save_trk_file = open(save_trk_file, "w")
    #     for result_tmp in results[hypo]:  # N x 15
    #         save_results(
    #             result_tmp,
    #             save_trk_file,
    #             eval_file_dict[hypo],
    #             det_id2str,
    #             frame,
    #             cfg.score_threshold,
    #         )
    #     save_trk_file.close()

    # dets += 1
    # seq_count += 1

    for index in range(cfg.num_hypo):
        # eval_file_dict[index].close()
        ID_start = max(ID_start, tracker.ID_count[index])

    # print_log(
    #     "%s, %25s: %4.f seconds for %5d frames or %6.1f FPS, metric is %s = %.2f"
    #     % (
    #         cfg.dataset,
    #         result_sha,
    #         total_time,
    #         total_frames,
    #         total_frames / total_time,
    #         tracker.metric,
    #         tracker.thres,
    #     ),
    #     # log=log,
    # )

    return ID_start


def main():

    # load config files
    config_path = "./configs/config.yml"
    cfg = Config(config_path)

    # global ID counter used for all categories, not start from 1 for each category to prevent different
    # categories of objects have the same ID. This allows visualization of all object categories together
    # without ID conflicting, Also use 1 (not 0) as start because MOT benchmark requires positive ID
    ID_start = 1

    # run tracking for each category
    for cla in cfg.class_list:
        ID_start = main_per_cat(cfg, cla, ID_start)


if __name__ == "__main__":
    main()

