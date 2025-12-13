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

    @org.springframework.beans.factory.annotation.Autowired
    private org.springframework.context.ConfigurableApplicationContext context;

    @org.springframework.beans.factory.annotation.Autowired
    private org.springframework.boot.ApplicationArguments appArgs;

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
                    boolean runVerification = "true".equalsIgnoreCase(System.getProperty("verifyFeedbackNoLeakage"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackNoLeakage") && "true"
                                    .equalsIgnoreCase(appArgs.getOptionValues("verifyFeedbackNoLeakage").get(0)));

                    boolean runSweep = "true".equalsIgnoreCase(System.getProperty("verifyFeedbackSweep"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackSweep")
                                    && "true".equalsIgnoreCase(appArgs.getOptionValues("verifyFeedbackSweep").get(0)));

                    if (runSweep) {
                        runFeedbackSweep(model, trainingData, testingData, patch4x4Model);
                    } else if (runVerification) {
                        runLeakageFreeVerification(model, testingData, trainingData, patch4x4Model);
                    } else {
                        // Legacy Evaluation
                        model.evaluateAccuracy(testingData);

                        // MRF Evaluation
                        try (InputStream is = getClass().getResourceAsStream("/mrf_config.json")) {
                            if (is != null) {
                                FactorGraphBuilder builder = new FactorGraphBuilder(
                                        model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                                        model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                                        patch4x4Model);

                                ObjectMapper mapper = new ObjectMapper();
                                FactorGraphBuilder.ConfigRoot config = mapper.readValue(
                                        getClass().getResourceAsStream("/mrf_config.json"),
                                        FactorGraphBuilder.ConfigRoot.class);

                                Map<String, DigitFactorNode> nodes = builder
                                        .build(getClass().getResourceAsStream("/mrf_config.json"));
                                DigitFactorNode root = nodes.get(config.rootNodeId);

                                if (root != null) {
                                    MarkovFieldDigitClassifier mrf = new MarkovFieldDigitClassifier(root);
                                    // Default evaluation on test set (treat as test)
                                    mrf.evaluateAccuracy(testingData, true);
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
        String searchPattern = (pattern != null) ? pattern : "";
        Resource[] resources = resolver.getResources(searchPattern);

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

    private void runLeakageFreeVerification(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model) {
        logger.info("============================================================");
        logger.info("STARTING LEAKAGE-FREE VERIFICATION PROTOCOL");
        logger.info("============================================================");

        try {
            // 1. Build MRF
            FactorGraphBuilder builder = new FactorGraphBuilder(
                    model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                    model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                    patch4x4Model);

            ObjectMapper mapper = new ObjectMapper();
            FactorGraphBuilder.ConfigRoot configRoot = mapper.readValue(
                    getClass().getResourceAsStream("/mrf_config.json"),
                    FactorGraphBuilder.ConfigRoot.class);
            Map<String, DigitFactorNode> nodes = builder.build(getClass().getResourceAsStream("/mrf_config.json"));
            DigitFactorNode root = nodes.get(configRoot.rootNodeId);
            if (root == null)
                throw new RuntimeException("Root node not found");

            MarkovFieldDigitClassifier mrf = new MarkovFieldDigitClassifier(root);

            // Get base config from file
            com.markovai.server.ai.Patch4x4FeedbackConfig baseConfig = getBaseConfigFromFile();

            // 2. Phase A: Baseline (Feedback Disabled)
            logger.info("PHASE A: Baseline Test Accuracy (Feedback Disabled)");
            com.markovai.server.ai.Patch4x4FeedbackConfig baselineCfg = baseConfig.copy();
            baselineCfg.enabled = false;
            mrf.setPatch4x4Config(baselineCfg);
            mrf.evaluateAccuracy(testData, true);

            // 3. Phase B: Adaptation (Feedback Enabled, Learning Enabled) on TRAIN subset
            // Deterministic sampling
            int adaptSize = 2000;
            List<DigitImage> adaptSet = selectAdaptationSubset(trainData, adaptSize, 12345L);
            logger.info("PHASE B: Adaptation on TRAIN subset ({} images). Feedback Learning ENABLED.", adaptSet.size());

            // Enable feedback and learning
            com.markovai.server.ai.Patch4x4FeedbackConfig adaptationCfg = baseConfig.copy();
            adaptationCfg.enabled = true;
            adaptationCfg.learningEnabled = true;
            mrf.setPatch4x4Config(adaptationCfg);

            // Reset MRF state before adaptation (just in case, though it's fresh here)
            // But we actually want to ensure the node state is clean. The MRF was just
            // built, so it is clean.

            // Run evaluation on ADAPT set (isTestSet=false)
            mrf.evaluateAccuracy(adaptSet, false);
            logger.info("Adaptation complete.");

            // 4. Phase C: Final Test (Feedback Enabled, Learning FROZEN)
            logger.info("PHASE C: Final Test Accuracy (Feedback Scoring Enabled, Learning Frozen)");

            com.markovai.server.ai.Patch4x4FeedbackConfig finalCfg = baseConfig.copy();
            finalCfg.enabled = true;
            finalCfg.learningEnabled = false;
            mrf.setPatch4x4Config(finalCfg);

            // Run on Test Data (isTestSet=true)
            mrf.evaluateAccuracy(testData, true);

            logger.info("============================================================");
            logger.info("VERIFICATION PROTOCOL COMPLETE");
            logger.info("============================================================");

            // Force shutdown since we are in a CLI-like verification mode
            if (context != null) {
                logger.info("Shutting down application...");
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Verification failed", e);
        }
    }

    private void runFeedbackSweep(RowColumnDigitClassifier model, List<DigitImage> trainData, List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model) {
        logger.info("============================================================");
        logger.info("STARTING FEEDBACK HYPERPARAMETER SWEEP");
        logger.info("============================================================");

        try {
            // 1. Compute Baseline (Once)
            logger.info("Computing Baseline Accuracy...");
            double baselineAcc = evaluateSweepBaseline(model, testData, patch4x4Model);
            logger.info("Baseline Accuracy: {}", String.format("%.4f", baselineAcc));

            // Explicit Grid
            double[] adjScales = { 0.10, 0.05, 0.02, 0.01 };
            double[] etas = { 0.003, 0.001, 0.0005 };

            List<SweepResult> results = new ArrayList<>();

            for (double scale : adjScales) {
                for (double eta : etas) {
                    logger.info("Running sweep iteration: adjScale={}, eta={}", scale, eta);
                    // MRF is rebuilt every iteration, effectively resetting node state.

                    double adaptedAcc = runSweepIteration(model, trainData, testData, patch4x4Model, scale, eta);

                    results.add(new SweepResult(scale, eta, baselineAcc, adaptedAcc));
                }
            }

            // Print Full Table
            System.out.println("\n=== FULL LEAKAGE-FREE FEEDBACK SWEEP RESULTS ===");
            System.out.println("freqScalingEnabled=true");
            System.out.println("freqScalingMode=GLOBAL_SQRT");
            System.out.println("decayEveryNUpdates=5000");
            System.out.println("adaptationSetSize=2000");
            System.out.println();
            System.out.println("adjScale\teta\tbaselineAcc\tadaptedAcc\tdelta");
            System.out.println("-------------------------------------------------------");

            for (SweepResult r : results) {
                System.out.printf("%.4f\t%.4f\t%.4f\t%.4f\t%.4f%n",
                        r.adjScale, r.eta, r.baselineAcc, r.adaptedAcc, r.getDelta());
            }

            // CSV Output
            boolean printCsv = "true".equalsIgnoreCase(System.getProperty("printSweepAsCSV")) ||
                    (appArgs != null && appArgs.containsOption("printSweepAsCSV")
                            && "true".equalsIgnoreCase(appArgs.getOptionValues("printSweepAsCSV").get(0)));

            if (printCsv) {
                System.out.println("\n=== CSV OUTPUT ===");
                System.out.println("adjScale,eta,baselineAcc,adaptedAcc,delta");
                for (SweepResult r : results) {
                    System.out.printf("%.4f,%.4f,%.4f,%.4f,%.4f%n",
                            r.adjScale, r.eta, r.baselineAcc, r.adaptedAcc, r.getDelta());
                }
            }

            logger.info("============================================================");
            logger.info("SWEEP COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Sweep failed", e);
        }
    }

    private static class SweepResult {
        double adjScale;
        double eta;
        double baselineAcc;
        double adaptedAcc;

        public SweepResult(double adjScale, double eta, double baselineAcc, double adaptedAcc) {
            this.adjScale = adjScale;
            this.eta = eta;
            this.baselineAcc = baselineAcc;
            this.adaptedAcc = adaptedAcc;
        }

        public double getDelta() {
            return adaptedAcc - baselineAcc;
        }
    }

    private double evaluateSweepBaseline(RowColumnDigitClassifier model, List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model) throws Exception {
        MarkovFieldDigitClassifier mrf = buildMrf(model, patch4x4Model);
        mrf.setPatch4x4Config(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
        return mrf.evaluateAccuracy(testData, true);
    }

    private double runSweepIteration(RowColumnDigitClassifier model, List<DigitImage> trainData,
            List<DigitImage> testData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model, double adjScale,
            double eta) throws Exception {

        // Build Fresh MRF
        MarkovFieldDigitClassifier mrf = buildMrf(model, patch4x4Model);

        // Get Base Config (so we respect file settings for other params)
        com.markovai.server.ai.Patch4x4FeedbackConfig baseConfig = getBaseConfigFromFile();

        // Adaptation Train Subset (2000) - Deterministic
        List<DigitImage> adaptSet = selectAdaptationSubset(trainData, 2000, 12345L);

        // Enable Feedback (Adaptation Mode)
        com.markovai.server.ai.Patch4x4FeedbackConfig adaptationCfg = baseConfig.copy();
        adaptationCfg.enabled = true;
        adaptationCfg.learningEnabled = true;
        adaptationCfg.adjScale = adjScale; // Override from sweep
        adaptationCfg.eta = eta; // Override from sweep

        mrf.setPatch4x4Config(adaptationCfg);
        mrf.evaluateAccuracy(adaptSet, false);

        // Freeze and Test
        com.markovai.server.ai.Patch4x4FeedbackConfig finalCfg = adaptationCfg.copy();
        finalCfg.learningEnabled = false;
        // Keep adjScale / eta same for inference (scoring)

        mrf.setPatch4x4Config(finalCfg);

        return mrf.evaluateAccuracy(testData, true);
    }

    private MarkovFieldDigitClassifier buildMrf(RowColumnDigitClassifier model,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model) throws Exception {
        FactorGraphBuilder builder = new FactorGraphBuilder(
                model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                patch4x4Model);

        FactorGraphBuilder.ConfigRoot configRoot = loadMrfConfig();
        Map<String, DigitFactorNode> nodes = builder.build(getClass().getResourceAsStream("/mrf_config.json"));
        DigitFactorNode root = nodes.get(configRoot.rootNodeId);
        if (root == null)
            throw new RuntimeException("Root node not found");

        return new MarkovFieldDigitClassifier(root);
    }

    private FactorGraphBuilder.ConfigRoot loadMrfConfig() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(
                getClass().getResourceAsStream("/mrf_config.json"),
                FactorGraphBuilder.ConfigRoot.class);
    }

    /**
     * Finds the base configuration from the loaded file.
     */
    private com.markovai.server.ai.Patch4x4FeedbackConfig getBaseConfigFromFile() throws Exception {
        FactorGraphBuilder.ConfigRoot root = loadMrfConfig();
        for (FactorGraphBuilder.ConfigNode node : root.nodes) {
            if ("Patch4x4Node".equals(node.type) && node.feedback != null) {
                return node.feedback;
            }
        }
        // Fallback default if not found in JSON
        logger.warn("Patch4x4Node feedback config not found in JSON, using defaults.");
        return new com.markovai.server.ai.Patch4x4FeedbackConfig(
                false, false, 0.10, 0.003, 0.02, true, true, 5.0, false, 1.0e-4,
                false, "GLOBAL_SQRT", 0.05, 1.0, 0);
    }

    private List<DigitImage> selectAdaptationSubset(List<DigitImage> trainData, int adaptSize, long seed) {
        List<DigitImage> copy = new ArrayList<>(trainData);
        java.util.Collections.shuffle(copy, new java.util.Random(seed));
        return copy.subList(0, Math.min(adaptSize, copy.size()));
    }
}
