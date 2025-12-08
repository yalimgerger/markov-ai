package com.markovai.server.service;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.RowColumnDigitClassifier;
import com.markovai.server.ai.MarkovFieldDigitClassifier;
import com.markovai.server.ai.hierarchy.DigitFactorNode;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import jakarta.annotation.PostConstruct;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class MarkovTrainingService {

    private static final Logger logger = LoggerFactory.getLogger(MarkovTrainingService.class);
    private final RowColumnDigitClassifier model = new RowColumnDigitClassifier();
    private final com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model = new com.markovai.server.ai.DigitPatch4x4UnigramModel();
    private boolean isReady = false;

    public RowColumnDigitClassifier getModel() {
        return model;
    }

    public boolean isReady() {
        return isReady;
    }

    @PostConstruct
    public void init() {
        new Thread(() -> {
            try {
                logger.info("Initializing Markov Training Service...");
                List<DigitImage> trainingData = loadImages("classpath:mnist/training/*/*.png");
                List<DigitImage> testingData = loadImages("classpath:mnist/testing/*/*.png");

                if (trainingData.isEmpty()) {
                    logger.error("No training data found!");
                    return;
                }

                model.train(trainingData);

                // Train 4x4 Unigram Model
                logger.info("Training 4x4 Patch Model...");
                for (DigitImage img : trainingData) {
                    int[][] binary = com.markovai.server.ai.DigitMarkovModel.binarize(img.pixels, 128);
                    patch4x4Model.trainOnImage(img.label, binary);
                }
                patch4x4Model.finalizeProbabilities();
                logger.info("4x4 Patch Model Trained.");

                if (!testingData.isEmpty()) {
                    // Legacy Evaluation
                    model.evaluateAccuracy(testingData);

                    // MRF Evaluation
                    try (InputStream is = getClass().getResourceAsStream("/mrf_config.json")) {
                        if (is != null) {
                            FactorGraphBuilder builder = new FactorGraphBuilder(
                                    model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                                    model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                                    patch4x4Model);

                            // Naive config root resolution
                            // Re-parsing to find root ID or let Builder handle it?
                            // Builder returns Map<String, Node>, we need root ID.
                            // I'll update Builder to just return Root or a Result object, or parse JSON
                            // separately?
                            // Actually Builder.build returns Map. I need to find the root.
                            // Let's reload config to get root ID or just ask Builder to return it.
                            // To keep it simple, I'll modify Builder to assume root is returned or
                            // accessible.
                            // Wait, I designed Builder to return Map. I can just parse config here to get
                            // root ID, or modify Builder.

                            // Better: Let's use the ObjectMapper here to get the root ID quickly
                            // OR update FactorGraphBuilder to return a graph object containing root.
                            // Since I can't easily change Builder signature in replace_file cleanly without
                            // seeing it,
                            // I'll assume I can read the JSON again or use a helper.

                            // Actually, I'll just change FactorGraphBuilder to return a Graph structure in
                            // a separate file or just parse locally.
                            // Let's stick with parsing here to get root ID.
                            ObjectMapper mapper = new ObjectMapper();
                            FactorGraphBuilder.ConfigRoot config = mapper.readValue(
                                    getClass().getResourceAsStream("/mrf_config.json"),
                                    FactorGraphBuilder.ConfigRoot.class);

                            Map<String, DigitFactorNode> nodes = builder
                                    .build(getClass().getResourceAsStream("/mrf_config.json"));
                            DigitFactorNode root = nodes.get(config.rootNodeId);

                            if (root != null) {
                                MarkovFieldDigitClassifier mrf = new MarkovFieldDigitClassifier(root, nodes);
                                mrf.evaluateAccuracy(testingData);
                            } else {
                                logger.error("MRF Root node not found: {}", config.rootNodeId);
                            }
                        } else {
                            logger.error("mrf_config.json not found!");
                        }
                    } catch (Exception ex) {
                        logger.error("Failed to init MRF", ex);
                    }
                }

                isReady = true;
                logger.info("Markov Model is ready for classification.");
            } catch (Exception e) {
                logger.error("Failed to initialize Markov model", e);
            }
        }).start();
    }

    private List<DigitImage> loadImages(String pattern) throws IOException {
        List<DigitImage> images = new ArrayList<>();
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources(pattern);

        logger.info("Found {} image files for pattern: {}", resources.length, pattern);

        for (Resource res : resources) {
            try (InputStream is = res.getInputStream()) {
                BufferedImage bi = ImageIO.read(is);
                if (bi == null)
                    continue;

                int width = bi.getWidth();
                int height = bi.getHeight();
                int[][] pixels = new int[height][width];

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        // Get grayscale value. PNGs are likely grayscale, but safely extract brightness
                        int clr = bi.getRGB(x, y);
                        int red = (clr & 0x00ff0000) >> 16;
                        int green = (clr & 0x0000ff00) >> 8;
                        int blue = clr & 0x000000ff;
                        // Simple average for grayscale
                        int gray = (red + green + blue) / 3;
                        pixels[y][x] = gray;
                    }
                }

                // Extract label from parent directory name
                // Format: .../mnist/training/5/img.png -> label is 5
                int label = -1;
                String relPath = null;
                try {
                    String path = res.getURL().getPath();
                    String[] parts = path.split("/");
                    // Expected structure ending: .../mnist/training/5/filename.png
                    // We want to extract e.g. "training/5/filename.png" or "testing/5/filename.png"
                    // Find "mnist" index and take substring?
                    // Or finding "training" or "testing"

                    int idx = -1;
                    for (int i = 0; i < parts.length; i++) {
                        if ("training".equals(parts[i]) || "testing".equals(parts[i])) {
                            idx = i;
                            break;
                        }
                    }

                    if (idx != -1 && idx < parts.length - 1) {
                        StringBuilder sb = new StringBuilder();
                        for (int i = idx; i < parts.length; i++) {
                            if (sb.length() > 0)
                                sb.append("/");
                            sb.append(parts[i]);
                        }
                        relPath = sb.toString();

                        // Label is parts[length-2]
                        label = Integer.parseInt(parts[parts.length - 2]);
                    } else {
                        // Fallback
                        label = Integer.parseInt(parts[parts.length - 2]);
                        relPath = "unknown/" + label + "/" + parts[parts.length - 1];
                    }

                } catch (Exception e) {
                    logger.warn("Could not determine metadata for {}", res.getDescription());
                }

                String hash = computeHash(pixels);

                images.add(new DigitImage(pixels, label, relPath, hash));
            } catch (Exception e) {
                logger.warn("Failed to load image: {}", res.getFilename(), e);
            }
        }
        return images;
    }

    private String computeHash(int[][] pixels) {
        try {
            // Binarize using same logic as main model
            int[][] binary = com.markovai.server.ai.DigitMarkovModel.binarize(pixels, 128);
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
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
