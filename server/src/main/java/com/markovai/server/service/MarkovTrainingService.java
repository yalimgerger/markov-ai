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

                if (!testingData.isEmpty()) {
                    // Legacy Evaluation
                    model.evaluateAccuracy(testingData);

                    // MRF Evaluation
                    try (InputStream is = getClass().getResourceAsStream("/mrf_config.json")) {
                        if (is != null) {
                            FactorGraphBuilder builder = new FactorGraphBuilder(
                                    model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                                    model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor());

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
                // Resource URL might be: file:/.../mnist/training/5/3423.png
                // We can parse the path.
                int label = extractLabel(res);

                images.add(new DigitImage(pixels, label));
            } catch (Exception e) {
                logger.warn("Failed to load image: {}", res.getFilename(), e);
            }
        }
        return images;
    }

    private int extractLabel(Resource res) {
        try {
            // Typical path: .../training/5/123.png
            String path = res.getURL().getPath();
            String[] parts = path.split("/");
            // The label should be the directory name containing the file
            // parts[length-1] is filename
            // parts[length-2] is label directory
            String labelStr = parts[parts.length - 2];
            return Integer.parseInt(labelStr);
        } catch (Exception e) {
            logger.warn("Could not extract label from path {}, defaulting to -1", res.getDescription());
            return -1;
        }
    }
}
