# Pose-Guided Temporal Sign-Language Recognition

This project classifies isolated dynamic sign-language gestures from short clips. It extracts hand and upper-body landmarks with MediaPipe, normalizes them into variable-length temporal sequences, and trains a PyTorch bidirectional LSTM classifier with padding-aware sequence handling.

It is not a continuous sign-language translation system. It predicts one class for one isolated gesture clip.
