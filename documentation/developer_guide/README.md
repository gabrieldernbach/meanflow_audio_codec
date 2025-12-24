# Developer Guide

This section contains documentation for developers working on the Meanflow Audio Codec project.

## Contents

- [Architecture](architecture.md) - High-level system architecture and design
- [Code Organization](code_organization.md) - Project structure and organization patterns
- [API Reference](api_reference.md) - API documentation for key components
- [Contributing](contributing.md) - Guidelines for contributing to the project

## Overview

The Meanflow Audio Codec is a JAX/Flax implementation of an MDCT-based autoencoder audio codec using Improved Mean Flows. The codebase follows patterns inspired by Google Research's BigVision repository.

## Key Components

- **Models**: Flow models for audio encoding (`models/`)
- **Trainers**: Training loops and utilities (`trainers/`)
- **Evaluators**: Sampling and metrics (`evaluators/`)
- **Datasets**: Data loading (`datasets/`)
- **Preprocessing**: MDCT utilities (`preprocessing/`)
- **Configs**: Configuration dataclasses (`configs/`)

For detailed information, see the [Architecture](architecture.md) and [Code Organization](code_organization.md) documents.

