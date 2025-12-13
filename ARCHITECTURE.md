# Project Architecture

## Overview
This project is an AI application that uses Markov Chains and Markov Random Fields (MRF) for digit classification or generation (specifically on the MNIST dataset). It consists of a Java Spring Boot backend for the core AI logic and database interactions, and a Vite-based JavaScript frontend for the user interface.

## Technology Stack

### Backend (`server/`)
- **Language**: Java 21
- **Framework**: Spring Boot (Web)
- **Build Tool**: Gradle (Kotlin DSL)
- **Database**: SQLite (via unique JDBC implementation)
- **Logic**: Custom Markov Field implementation for digit classification.

### Frontend (`client/`)
- **Language**: JavaScript (ESM)
- **Bundler**: Vite
- **Styling**: Vanilla CSS (assumed based on current structure)
- **Integration**: Frontend is built and served as static resources by the Spring Boot server.

## Project Structure

- **`server/`**: Contains the backend code.
    - `src/main/java/com/markovai/server/`: Source code.
        - `MarkovAiApplication.java`: Main entry point for the web server.
        - `tools/DigitDatasetPrecompute.java`: Offline tool for training/precomputing the Markov models.
    - `build.gradle.kts`: Gradle configuration. Includes tasks to build frontend and copy it to resources.
    - `markov_cache.db`: SQLite database storing precomputed models/chains.

- **`client/`**: Contains the frontend code.
    - `src/`: Source files.
    - `vite.config.js`: Vite configuration (likely proxies to backend or configures output dir).
    - `package.json`: NPM dependencies and scripts.

## Build & Run System

The project uses Gradle as the primary build orchestrator.

### Key Tasks
- **`./gradlew bootRun`**: Starts the Spring Boot application.
    - This automatically builds the frontend (via `npm run build` in `client/`) and serves it.
    - Access the app at `http://localhost:8080` (default Spring Boot port).
    - Includes a custom verification logic if `-DverifyFeedbackNoLeakage=true` is passed.
- **`./gradlew precompute`**: Runs the offline training tool (`DigitDatasetPrecompute`).
    - Used to populate the `markov_cache.db`.

### Frontend Integration
The `server/build.gradle.kts` file defines `npmInstall` and `npmBuild` tasks. `processResources` depends on `npmBuild`, ensuring that whenever the java server is built or run, the latest frontend code is compiled and included in the jar/classpath.

## Key Components

### `MarkovAiApplication`
The standard Spring Boot application class. It likely configures the web server and initializes the AI services using the data in `markov_cache.db`.

### `DigitDatasetPrecompute`
A command-line tool (run via Gradle) that processes the MNIST training data and builds the Markov chain models, saving them to the SQLite database. This separates the expensive training phase from the runtime serving phase.
