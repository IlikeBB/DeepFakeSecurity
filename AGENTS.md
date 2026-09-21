# 程式碼修改原則

- 採取最小必要修改，避免大幅改動程式碼或進行與任務無關的重構。
- 對已完善且運作正常的程式碼，除非任務確有需要，否則不要過度修改。
- 保持程式碼精簡、清楚且有效率，避免冗長實作、重複邏輯與不必要的複雜度。

# Project environment

Use the Conda environment `pt230` for this project's Python commands, dependency management, and tests.
Run commands with `conda run -n pt230 <command>`, or activate `pt230` before running them.

# Primary dataset

The primary dataset is located at `/ssd2/DeepFakes/celeb-df-video`.
It contains `Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/`, and `List_of_testing_videos.txt`.
Keep source dataset files unchanged; write generated features and other outputs separately.
