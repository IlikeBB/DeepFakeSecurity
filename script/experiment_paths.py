"""Canonical output directories for the two-stage retrieval experiment."""

from pathlib import Path


STAGE1_OUTPUT_NAME = "Stage1_Bank_Build"
STAGE1_CLUSTER_OUTPUT_NAME = "Stage1_Feature_Distribution"
STAGE2_OUTPUT_NAME = "Stage2_Anomaly_Evaluation"


def stage1_output(output):
    return Path(output) / STAGE1_OUTPUT_NAME


def stage1_cluster_output(output):
    return Path(output) / STAGE1_CLUSTER_OUTPUT_NAME


def stage2_output(output):
    return Path(output) / STAGE2_OUTPUT_NAME
