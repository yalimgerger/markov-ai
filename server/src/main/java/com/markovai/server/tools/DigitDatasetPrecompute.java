package com.markovai.server.tools;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.ai.DigitPatch4x4UnigramModel;
import com.markovai.server.ai.RowColumnDigitClassifier;
import com.markovai.server.ai.hierarchy.DigitFactorNode;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Offline tool to precompute Markov chain results and populate the SQLite
 * cache.
 * Usage: DigitDatasetPrecompute <datasetRoot>
 */
public class DigitDatasetPrecompute {

    private static final Logger logger = LoggerFactory.getLogger(DigitDatasetPrecompute.class);

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: DigitDatasetPrecompute <datasetRoot>");
            System.exit(1);
        }

        String datasetRoot = args[0];
        File rootDir = new File(datasetRoot);
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            System.err.println("Invalid dataset root: " + datasetRoot);
            System.exit(1);
        }

        logger.info("Starting Offline Precompute on {}", datasetRoot);

        // Force cleanup of old cache
        File dbFile = new File("markov_cache.db");
        if (dbFile.exists()) {
            logger.info("Deleting existing DB file to force recomputation: {}", dbFile.getAbsolutePath());
            if (!dbFile.delete()) {
                logger.error("Failed to delete existing DB file. Precompute might use stale data.");
            }
        }

        try {
            // 1. Load Images
            List<DigitImage> allImages = loadImagesFromFs(rootDir);
            logger.info("Loaded {} images from disk", allImages.size());

            if (allImages.isEmpty()) {
                logger.warn("No images found, exiting.");
                return;
            }

            // 2. Train Models (Strictly on Training Data)
            // Filter images to identify the training set.
            // Assuming standard MNIST directory structure: .../training/L/img.png vs
            // .../testing/L/img.png
            List<DigitImage> trainingImages = new ArrayList<>();
            for (DigitImage img : allImages) {
                if (img.imageRelPath != null && img.imageRelPath.contains("training")) {
                    trainingImages.add(img);
                }
            }
            logger.info("Identified {} training images out of {} total", trainingImages.size(), allImages.size());

            if (trainingImages.isEmpty()) {
                logger.warn("No training images identified! Models will be untrained.");
            }

            logger.info("Training models on training set only...");
            RowColumnDigitClassifier digitClassifier = new RowColumnDigitClassifier();
            digitClassifier.train(trainingImages);

            DigitPatch4x4UnigramModel patch4x4Model = new DigitPatch4x4UnigramModel();
            for (DigitImage img : trainingImages) {
                int[][] binary = DigitMarkovModel.binarize(img.pixels, 128);
                patch4x4Model.trainOnImage(img.label, binary);
            }
            patch4x4Model.finalizeProbabilities();
            logger.info("Models trained.");

            // 3. Build Factor Graph (Connects to DB)
            // We need to load mrf_config.json. Assuming it's in resources.
            // If running as tool, classpath might be tricky, but let's assume standard app
            // classpath.
            logger.info("Building Factor Graph / DB Access...");
            FactorGraphBuilder builder = new FactorGraphBuilder(
                    digitClassifier.getRowModel(),
                    digitClassifier.getColumnModel(),
                    digitClassifier.getPatchModel(),
                    digitClassifier.getRowExtractor(),
                    digitClassifier.getColumnExtractor(),
                    digitClassifier.getPatchExtractor(),
                    patch4x4Model);

            Map<String, DigitFactorNode> nodes = builder.build(
                    DigitDatasetPrecompute.class.getResourceAsStream("/mrf_config.json"));

            // 4. Compute and Cache
            logger.info("Precomputing chains for all images...");
            int count = 0;
            for (DigitImage img : allImages) {
                // Determine nodes to compute
                // We just trigger computeForImage on all nodes.
                // The cached nodes (Row, Col, Patch...) will check DB and write if missing.
                for (DigitFactorNode node : nodes.values()) {
                    // Skip WeightedSumNode as it requires children results and doesn't access DB
                    // itself.
                    // We only care about ensuring leaf nodes (Row, Col, Patch...) compute and cache
                    // their results.
                    if (node instanceof com.markovai.server.ai.hierarchy.WeightedSumNode) {
                        continue;
                    }
                    node.computeForImage(img, null);
                }

                count++;
                if (count % 100 == 0) {
                    logger.info("Processed {}/{}", count, allImages.size());
                }
            }
            logger.info("Precompute complete. Processed {} images.", count);

        } catch (Exception e) {
            logger.error("Precompute failed", e);
            System.exit(1);
        }
    }

    private static List<DigitImage> loadImagesFromFs(File rootDir) throws IOException {
        List<DigitImage> images = new ArrayList<>();
        Path rootPath = rootDir.toPath();

        Files.walkFileTree(rootPath, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (file.toString().endsWith(".png")) {
                    try {
                        BufferedImage bi = ImageIO.read(file.toFile());
                        if (bi != null) {
                            int width = bi.getWidth();
                            int height = bi.getHeight();
                            int[][] pixels = new int[height][width];
                            for (int y = 0; y < height; y++) {
                                for (int x = 0; x < width; x++) {
                                    int clr = bi.getRGB(x, y);
                                    int red = (clr & 0x00ff0000) >> 16;
                                    int green = (clr & 0x0000ff00) >> 8;
                                    int blue = clr & 0x000000ff;
                                    pixels[y][x] = (red + green + blue) / 3;
                                }
                            }

                            // Extract label: parent dir name
                            String labelStr = file.getParent().getFileName().toString();
                            int label = -1;
                            try {
                                label = Integer.parseInt(labelStr);
                            } catch (NumberFormatException ignored) {
                            }

                            // Extract relative path
                            String relPath = rootPath.relativize(file).toString();

                            // Compute Hash (SHA-256 of binary)
                            String hash = computeHash(pixels);

                            images.add(new DigitImage(pixels, label, relPath, hash));
                        }
                    } catch (Exception e) {
                        logger.warn("Failed to load {}", file, e);
                    }
                }
                return FileVisitResult.CONTINUE;
            }
        });

        return images;
    }

    private static String computeHash(int[][] pixels) {
        try {
            int[][] binary = DigitMarkovModel.binarize(pixels, 128);
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    digest.update((byte) binary[r][c]);
                }
            }
            byte[] hash = digest.digest();
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1)
                    hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (Exception e) {
            return null;
        }
    }
}
