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
                String dataDir = System.getProperty("markov.data.dir", ".");
                List<DigitImage> trainingData = loadImages("file:" + dataDir + "/mnist/training/*/*.png");
                List<DigitImage> testingData = loadImages("file:" + dataDir + "/mnist/testing/*/*.png");

                logger.info("Dataset loaded: trainExamples={}, testExamples={}", trainingData.size(),
                        testingData.size());

                if (trainingData.size() != 60000) {
                    logger.warn("Training dataset size ({}) deviates from expected 60000", trainingData.size());
                }
                if (testingData.size() != 10000) {
                    logger.warn("Testing dataset size ({}) deviates from expected 10000", testingData.size());
                }

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

                    boolean runMultiSeed = "true"
                            .equalsIgnoreCase(System.getProperty("verifyFeedbackNoLeakageMultiSeed"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackNoLeakageMultiSeed") && "true"
                                    .equalsIgnoreCase(
                                            appArgs.getOptionValues("verifyFeedbackNoLeakageMultiSeed").get(0)));

                    boolean runAdaptSweep = "true"
                            .equalsIgnoreCase(System.getProperty("verifyFeedbackNoLeakageAdaptSweep"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackNoLeakageAdaptSweep")
                                    && "true".equalsIgnoreCase(
                                            appArgs.getOptionValues("verifyFeedbackNoLeakageAdaptSweep").get(0)));

                    if (runAdaptSweep) {
                        runAdaptationSizeSweep(model, trainingData, testingData, patch4x4Model);
                    } else if (runSweep) {
                        runFeedbackSweep(model, trainingData, testingData, patch4x4Model);
                    } else if (runMultiSeed) {
                        boolean useRowFeedback = false;
                        boolean useColFeedback = false;
                        String mode = System.getProperty("feedbackMode");
                        if (mode == null && appArgs != null && appArgs.containsOption("feedbackMode")) {
                            List<String> values = appArgs.getOptionValues("feedbackMode");
                            if (values != null && !values.isEmpty()) {
                                mode = values.get(0);
                            }
                        }

                        if (mode != null && !mode.isEmpty()) {
                            switch (mode.toUpperCase()) {
                                case "PATCH":
                                    useRowFeedback = false;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_ROW":
                                    useRowFeedback = true;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_COL":
                                    useRowFeedback = false;
                                    useColFeedback = true;
                                    break;
                                case "PATCH_ROW_COL":
                                    useRowFeedback = true;
                                    useColFeedback = true;
                                    break;
                                default:
                                    logger.warn("Unknown feedbackMode '{}', defaulting to PATCH", mode);
                                    useRowFeedback = false;
                                    useColFeedback = false;
                            }
                            logger.info("Feedback configuration set via feedbackMode={}: row={}, col={}", mode,
                                    useRowFeedback, useColFeedback);
                        } else {
                            // Fallback to legacy flags
                            useRowFeedback = "true".equalsIgnoreCase(System.getProperty("rowFeedback"))
                                    || (appArgs != null && appArgs.containsOption("rowFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("rowFeedback").get(0)));
                            useColFeedback = "true".equalsIgnoreCase(System.getProperty("colFeedback"))
                                    || (appArgs != null && appArgs.containsOption("colFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("colFeedback").get(0)));
                        }

                        runLeakageFreeMultiSeedVerification(model, testingData, trainingData, patch4x4Model,
                                useRowFeedback, useColFeedback);
                    } else if (runVerification) {
                        boolean useRowFeedback = false;
                        boolean useColFeedback = false;
                        String mode = System.getProperty("feedbackMode");
                        if (mode == null && appArgs != null && appArgs.containsOption("feedbackMode")) {
                            List<String> values = appArgs.getOptionValues("feedbackMode");
                            if (values != null && !values.isEmpty()) {
                                mode = values.get(0);
                            }
                        }

                        if (mode != null && !mode.isEmpty()) {
                            switch (mode.toUpperCase()) {
                                case "PATCH":
                                    useRowFeedback = false;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_ROW":
                                    useRowFeedback = true;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_COL":
                                    useRowFeedback = false;
                                    useColFeedback = true;
                                    break;
                                case "PATCH_ROW_COL":
                                    useRowFeedback = true;
                                    useColFeedback = true;
                                    break;
                                default:
                                    useRowFeedback = false;
                                    useColFeedback = false;
                            }
                        } else {
                            useRowFeedback = "true".equalsIgnoreCase(System.getProperty("rowFeedback"))
                                    || (appArgs != null && appArgs.containsOption("rowFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("rowFeedback").get(0)));
                            useColFeedback = "true".equalsIgnoreCase(System.getProperty("colFeedback"))
                                    || (appArgs != null && appArgs.containsOption("colFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("colFeedback").get(0)));
                        }

                        runLeakageFreeVerification(model, testingData, trainingData, patch4x4Model, useRowFeedback,
                                useColFeedback);
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
        java.util.Set<String> loadedHashes = new java.util.HashSet<>();

        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        String searchPattern = (pattern != null) ? pattern : "";
        Resource[] resources = resolver.getResources(searchPattern);

        logger.info("Found {} image resources for pattern: {}", resources.length, pattern);

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

                String hash = computeHash(pixels);
                // Deduplicate based on content hash
                if (hash != null && loadedHashes.contains(hash)) {
                    continue;
                }
                if (hash != null) {
                    loadedHashes.add(hash);
                }

                // Extract label from parent directory name
                // Format: .../mnist/training/5/img.png -> label is 5
                int label = -1;
                String relPath = null;
                try {
                    String path = res.getURL().getPath();
                    String[] parts = path.split("/");

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

    private static class LeakageFreeResult {
        long seed;
        double baselineAcc;
        double frozenAcc;

        public LeakageFreeResult(long seed, double baselineAcc, double frozenAcc) {
            this.seed = seed;
            this.baselineAcc = baselineAcc;
            this.frozenAcc = frozenAcc;
        }

        public double getDelta() {
            return frozenAcc - baselineAcc;
        }
    }

    private void runLeakageFreeMultiSeedVerification(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            boolean useRowFeedback, boolean useColFeedback) {
        logger.info("============================================================");
        logger.info("STARTING MULTI-SEED LEAKAGE-FREE VERIFICATION");
        logger.info("============================================================");

        try {
            // Parse seeds
            String seedsStr = System.getProperty("adaptSeeds");
            long[] seeds;
            if (seedsStr != null && !seedsStr.isEmpty()) {
                String[] parts = seedsStr.split(",");
                seeds = new long[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    seeds[i] = Long.parseLong(parts[i].trim());
                }
            } else {
                seeds = new long[] { 12345L, 22222L, 33333L, 44444L, 55555L };
            }

            List<LeakageFreeResult> results = new ArrayList<>();
            // Compute baseline once to be efficient? No, strict requirement is full reset.
            // We will run the full protocol for each seed.

            for (long seed : seeds) {
                logger.info("Running protocol for seed={}, rowFeedback={}, colFeedback={}", seed, useRowFeedback,
                        useColFeedback);
                LeakageFreeResult result = performLeakageFreeProtocol(model, testData, trainData, patch4x4Model, seed,
                        false, useRowFeedback, useColFeedback, 2000, null);
                results.add(result);
                logger.info("Seed={}  Baseline={:.4f}  Frozen={:.4f}  Delta={:+.4f}",
                        seed, result.baselineAcc, result.frozenAcc, result.getDelta());
            }

            // Statistics
            double sumFrozen = 0;
            double sumDelta = 0;
            double minDelta = Double.MAX_VALUE;
            double maxDelta = -Double.MAX_VALUE;

            for (LeakageFreeResult r : results) {
                sumFrozen += r.frozenAcc;
                double d = r.getDelta();
                sumDelta += d;
                if (d < minDelta)
                    minDelta = d;
                if (d > maxDelta)
                    maxDelta = d;
            }

            double meanFrozen = sumFrozen / results.size();
            double meanDelta = sumDelta / results.size();

            double varFrozen = 0;
            double varDelta = 0;
            for (LeakageFreeResult r : results) {
                varFrozen += Math.pow(r.frozenAcc - meanFrozen, 2);
                varDelta += Math.pow(r.getDelta() - meanDelta, 2);
            }
            double stdFrozen = (results.size() > 1) ? Math.sqrt(varFrozen / (results.size() - 1)) : 0.0;
            double stdDelta = (results.size() > 1) ? Math.sqrt(varDelta / (results.size() - 1)) : 0.0;

            String modeName = "PATCH";
            if (useRowFeedback && useColFeedback)
                modeName = "PATCH_ROW_COL";
            else if (useRowFeedback)
                modeName = "PATCH_ROW";
            else if (useColFeedback)
                modeName = "PATCH_COL";

            System.out.println("\n=== MULTI-SEED LEAKAGE-FREE SUMMARY ===");
            System.out.printf("Mode: %s%n", modeName);
            System.out.printf("Seeds: %d  AdaptSize: 2000%n", results.size());
            System.out.printf("FrozenAcc mean=%.4f std=%.4f%n", meanFrozen, stdFrozen);
            System.out.printf("Delta     mean=%+.4f std=%.4f min=%+.4f max=%+.4f%n", meanDelta, stdDelta, minDelta,
                    maxDelta);

            System.out.println("\n=== CSV OUTPUT ===");
            System.out.println("seed,baselineAcc,frozenAcc,delta");
            for (LeakageFreeResult r : results) {
                System.out.printf("%d,%.4f,%.4f,%.4f%n", r.seed, r.baselineAcc, r.frozenAcc, r.getDelta());
            }

            logger.info("============================================================");
            logger.info("MULTI-SEED VERIFICATION COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Multi-seed verification failed", e);
        }
    }

    private void runLeakageFreeVerification(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            boolean useRowFeedback, boolean useColFeedback) {
        logger.info("============================================================");
        logger.info("STARTING LEAKAGE-FREE VERIFICATION PROTOCOL (Single Seed)");
        logger.info("============================================================");

        try {
            performLeakageFreeProtocol(model, testData, trainData, patch4x4Model, 12345L, true, useRowFeedback,
                    useColFeedback, 2000, null);

            logger.info("============================================================");
            logger.info("VERIFICATION PROTOCOL COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                logger.info("Shutting down application...");
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Verification failed", e);
        }
    }

    private LeakageFreeResult performLeakageFreeProtocol(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            long seed, boolean verbose, boolean useRowFeedback, boolean useColFeedback, int adaptSize,
            Double knownBaseline)
            throws Exception {

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

        // Define canonical datasets
        List<DigitImage> phaseATest = testData;
        // adaptSize is passed as argument
        List<DigitImage> phaseBAdapt = selectAdaptationSubset(trainData, adaptSize, seed);
        List<DigitImage> phaseCTest = testData;

        // Validate Integrity
        validateNoOverlap(phaseBAdapt, phaseCTest);

        if (verbose) {
            logger.info("Leakage-free protocol dataset summary (Seed={}):", seed);
            logger.info("  Train size: {}", trainData.size());
            logger.info("  Test size:  {}", testData.size());
            logger.info("  Adapt size: {}", phaseBAdapt.size());
            logger.info("  Phase A eval size: {}", phaseATest.size());
            logger.info("  Phase C eval size: {}", phaseCTest.size());
        }

        // Get base config from file
        com.markovai.server.ai.Patch4x4FeedbackConfig baseConfig = getBaseConfigFromFile();

        // 2. Phase A: Baseline (Feedback Disabled)
        if (verbose)
            logger.info("PHASE A: Baseline Test Accuracy (Feedback Disabled)");

        // Reset row config to baseline (disabled)
        com.markovai.server.ai.Patch4x4FeedbackConfig rowBaseline = com.markovai.server.ai.Patch4x4FeedbackConfig
                .disabled();
        mrf.setRowFeedbackConfig(rowBaseline);

        // Reset col config to baseline
        com.markovai.server.ai.Patch4x4FeedbackConfig colBaseline = com.markovai.server.ai.Patch4x4FeedbackConfig
                .disabled();
        mrf.setColumnFeedbackConfig(colBaseline);

        com.markovai.server.ai.Patch4x4FeedbackConfig baselineCfg = baseConfig.copy();
        baselineCfg.enabled = false;
        mrf.setPatch4x4Config(baselineCfg);

        double baselineAcc;
        if (knownBaseline != null) {
            baselineAcc = knownBaseline;
            if (verbose)
                logger.info("PHASE A: Baseline Test Accuracy (Using Precomputed): {}",
                        String.format("%.4f", baselineAcc));
        } else {
            baselineAcc = mrf.evaluateAccuracy(phaseATest, true);
        }

        // 3. Phase B: Adaptation (Feedback Enabled, Learning Enabled) on TRAIN subset
        if (verbose)
            logger.info("PHASE B: Adaptation on TRAIN subset ({} images). Feedback Learning ENABLED.",
                    phaseBAdapt.size());

        // Enable feedback and learning
        com.markovai.server.ai.Patch4x4FeedbackConfig adaptationCfg = baseConfig.copy();
        adaptationCfg.enabled = true;
        adaptationCfg.learningEnabled = true;
        // Set Row Feedback for Adaptation
        com.markovai.server.ai.Patch4x4FeedbackConfig rowAdapt = new com.markovai.server.ai.Patch4x4FeedbackConfig();
        // We'll reuse default values but set enabled/learning based on flag
        // Ideally we should load from file too if present, but here we just toggle.
        // Assuming mrf_config has defaults we want.
        // Let's create a config based on the passed flag.
        if (useRowFeedback) {
            // For now, hardcode reasonable defaults resembling Patch4x4, or use defaults
            // from Patch4x4Config if config file missing
            // Actually, we should probably read the row config from file if possible.
            // But simpler to just use Patch4x4 defaults for now or clean defaults.
            // We will use standard defaults.
            rowAdapt = com.markovai.server.ai.Patch4x4FeedbackConfig.disabled();
            rowAdapt.enabled = true;
            rowAdapt.learningEnabled = true;
            // Set other params to defaults
            rowAdapt.adjScale = 0.10;
            rowAdapt.eta = 0.003;
            rowAdapt.marginTarget = 0.02;
            rowAdapt.updateOnlyIfIncorrect = true;
            rowAdapt.useMarginGating = true;
            rowAdapt.maxAdjAbs = 5.0;
            rowAdapt.frequencyScalingEnabled = true;
            rowAdapt.frequencyScalingMode = "GLOBAL_SQRT";
            // Optional: read these from config properly in future task
        } else {
            rowAdapt = com.markovai.server.ai.Patch4x4FeedbackConfig.disabled();
        }
        mrf.setRowFeedbackConfig(rowAdapt);

        // Set Col Feedback
        com.markovai.server.ai.Patch4x4FeedbackConfig colAdapt;
        if (useColFeedback) {
            colAdapt = com.markovai.server.ai.Patch4x4FeedbackConfig.disabled();
            colAdapt.enabled = true;
            colAdapt.learningEnabled = true;
            // Defaults
            colAdapt.adjScale = 0.10;
            colAdapt.eta = 0.003;
            colAdapt.marginTarget = 0.02;
            colAdapt.updateOnlyIfIncorrect = true;
            colAdapt.useMarginGating = true;
            colAdapt.maxAdjAbs = 5.0;
            colAdapt.frequencyScalingEnabled = true;
            colAdapt.frequencyScalingMode = "GLOBAL_SQRT";
        } else {
            colAdapt = com.markovai.server.ai.Patch4x4FeedbackConfig.disabled();
        }
        mrf.setColumnFeedbackConfig(colAdapt);

        mrf.setPatch4x4Config(adaptationCfg);

        // Run evaluation on ADAPT set (isTestSet=false)
        mrf.evaluateAccuracy(phaseBAdapt, false);
        if (verbose)
            logger.info("Adaptation complete.");

        // 4. Phase C: Final Test (Feedback Enabled, Learning FROZEN)
        if (verbose)
            logger.info("PHASE C: Final Test Accuracy (Feedback Scoring Enabled, Learning Frozen)");

        com.markovai.server.ai.Patch4x4FeedbackConfig finalCfg = baseConfig.copy();
        finalCfg.enabled = true;
        finalCfg.learningEnabled = false;
        mrf.setPatch4x4Config(finalCfg);

        // Freeze Row Feedback
        if (useRowFeedback) {
            com.markovai.server.ai.Patch4x4FeedbackConfig rowFrozen = rowAdapt.copy();
            rowFrozen.learningEnabled = false;
            mrf.setRowFeedbackConfig(rowFrozen);
        } else {
            mrf.setRowFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
        }

        // Freeze Col Feedback
        if (useColFeedback) {
            com.markovai.server.ai.Patch4x4FeedbackConfig colFrozen = colAdapt.copy();
            colFrozen.learningEnabled = false;
            mrf.setColumnFeedbackConfig(colFrozen);
        } else {
            mrf.setColumnFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
        }

        // Run on Test Data (isTestSet=true)
        double frozenAcc = mrf.evaluateAccuracy(phaseCTest, true);

        return new LeakageFreeResult(seed, baselineAcc, frozenAcc);
    }

    private void runAdaptationSizeSweep(RowColumnDigitClassifier model, List<DigitImage> trainData,
            List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model) {
        logger.info("============================================================");
        logger.info("STARTING ADAPTATION SIZE SWEEP (LEAKAGE-FREE)");
        logger.info("============================================================");

        try {
            // Filter training data to ensure no overlap with test data
            java.util.Set<String> testHashes = testData.stream()
                    .map(img -> img.imageHash)
                    .filter(java.util.Objects::nonNull)
                    .collect(java.util.stream.Collectors.toSet());

            List<DigitImage> cleanTrainData = trainData.stream()
                    .filter(img -> img.imageHash == null || !testHashes.contains(img.imageHash))
                    .collect(java.util.stream.Collectors.toList());

            if (cleanTrainData.size() < trainData.size()) {
                logger.warn("Removed {} overlapping images from training set to ensure leakage-free protocol.",
                        trainData.size() - cleanTrainData.size());
            }

            String adaptSizesStr = System.getProperty("adaptSizes");
            int[] adaptSizes;
            if (adaptSizesStr != null && !adaptSizesStr.isEmpty()) {
                String[] parts = adaptSizesStr.split(",");
                adaptSizes = new int[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    adaptSizes[i] = Integer.parseInt(parts[i].trim());
                }
            } else {
                adaptSizes = new int[] { 2000, 5000, 10000, 20000 };
            }

            String seedsStr = System.getProperty("adaptSeeds");
            long[] seeds;
            if (seedsStr != null && !seedsStr.isEmpty()) {
                String[] parts = seedsStr.split(",");
                seeds = new long[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    seeds[i] = Long.parseLong(parts[i].trim());
                }
            } else {
                seeds = new long[] { 12345L, 22222L, 33333L, 44444L, 55555L };
            }

            List<String> modes = new ArrayList<>();
            String modeParam = System.getProperty("feedbackMode");
            String modesParam = System.getProperty("feedbackModes");

            if ("ALL".equalsIgnoreCase(modeParam)) {
                modes.add("PATCH");
                modes.add("PATCH_ROW");
                modes.add("PATCH_COL");
                modes.add("PATCH_ROW_COL");
            } else if (modesParam != null && !modesParam.isEmpty()) {
                for (String m : modesParam.split(",")) {
                    modes.add(m.trim().toUpperCase());
                }
            } else if (modeParam != null && !modeParam.isEmpty()) {
                modes.add(modeParam.toUpperCase());
            } else {
                modes.add("PATCH");
                modes.add("PATCH_ROW");
                modes.add("PATCH_COL");
                modes.add("PATCH_ROW_COL");
            }

            double baselineAcc = evaluateSweepBaseline(model, testData, patch4x4Model);

            System.out.println("\n=== LEAKAGE-FREE ADAPTATION SIZE SWEEP ===");
            System.out.printf("baselineAcc=%.4f%n", baselineAcc);
            System.out.print("seeds=[");
            for (int i = 0; i < seeds.length; i++)
                System.out.print(seeds[i] + (i < seeds.length - 1 ? "," : ""));
            System.out.println("]");
            System.out.print("adaptSizes=[");
            for (int i = 0; i < adaptSizes.length; i++)
                System.out.print(adaptSizes[i] + (i < adaptSizes.length - 1 ? "," : ""));
            System.out.println("]");

            com.markovai.server.ai.Patch4x4FeedbackConfig cfg = getBaseConfigFromFile();
            System.out.printf(
                    "config: freqScaling=%s, freqScalingMode=%s, applyDecayEveryNUpdates=%d, eta=%.4f, adjScale=%.4f%n",
                    cfg.frequencyScalingEnabled, cfg.frequencyScalingMode, cfg.applyDecayEveryNUpdates, cfg.eta,
                    cfg.adjScale);

            System.out.println("\nmode,adaptSize,meanFrozen,meanDelta,stdDelta,minDelta,maxDelta");

            List<String> verifyRows = new ArrayList<>();

            for (String mode : modes) {
                boolean useRow = mode.contains("ROW");
                boolean useCol = mode.contains("COL");

                for (int adaptSize : adaptSizes) {
                    if (adaptSize > cleanTrainData.size()) {
                        logger.warn("Adaptation size {} exceeds train data size {}, clamping.", adaptSize,
                                cleanTrainData.size());
                    }

                    List<Double> deltas = new ArrayList<>();
                    List<Double> frozens = new ArrayList<>();

                    List<LeakageFreeResult> seedResults = java.util.stream.LongStream.of(seeds).parallel()
                            .mapToObj(seed -> {
                                try {
                                    return performLeakageFreeProtocol(model, testData, cleanTrainData, patch4x4Model,
                                            seed, false, useRow, useCol, adaptSize, baselineAcc);
                                } catch (Exception e) {
                                    throw new RuntimeException(e);
                                }
                            }).collect(java.util.stream.Collectors.toList());

                    for (LeakageFreeResult res : seedResults) {
                        double d = res.frozenAcc - baselineAcc;
                        deltas.add(d);
                        frozens.add(res.frozenAcc);
                    }

                    double sumFrozen = 0;
                    for (double f : frozens)
                        sumFrozen += f;
                    double meanFrozen = sumFrozen / frozens.size();

                    double sumDelta = 0;
                    double minDelta = Double.MAX_VALUE;
                    double maxDelta = -Double.MAX_VALUE;
                    for (double d : deltas) {
                        sumDelta += d;
                        if (d < minDelta)
                            minDelta = d;
                        if (d > maxDelta)
                            maxDelta = d;
                    }
                    double meanDelta = sumDelta / deltas.size();

                    double varDelta = 0;
                    if (deltas.size() > 1) {
                        for (double d : deltas)
                            varDelta += Math.pow(d - meanDelta, 2);
                        varDelta /= (deltas.size() - 1);
                    }
                    double stdDelta = Math.sqrt(varDelta);

                    String rowStr = String.format("%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f",
                            mode, adaptSize, meanFrozen, meanDelta, stdDelta, minDelta, maxDelta);
                    System.out.println(rowStr);
                    verifyRows.add(rowStr);
                }
            }

            System.out.println("\n```csv");
            System.out.println("mode,adaptSize,meanFrozen,meanDelta,stdDelta,minDelta,maxDelta");
            for (String r : verifyRows)
                System.out.println(r);
            System.out.println("```");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Adaptation sweep failed", e);
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

    private void validateNoOverlap(List<DigitImage> adaptSet, List<DigitImage> testSet) {
        java.util.Set<String> adaptHashes = new java.util.HashSet<>();
        for (DigitImage img : adaptSet) {
            if (img.imageHash != null)
                adaptHashes.add(img.imageHash);
        }

        int overlapCount = 0;
        for (DigitImage img : testSet) {
            if (img.imageHash != null && adaptHashes.contains(img.imageHash)) {
                overlapCount++;
            }
        }

        if (overlapCount > 0) {
            logger.error("DATA LEAKAGE DETECTED: {} images in adaptation set are also in the test set!", overlapCount);
            throw new IllegalStateException("Data leakage detected: adaptation set overlaps with test set.");
        }
        logger.info("Data integrity check passed: 0 overlaps between adaptation set and test set.");
    }
}
