package com.markovai.db;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.sql.SQLException;
import java.util.Optional;

public class PersistenceIntegrationTest {

    private static final String TEST_DB = "test_cache.db";
    private DigitImageDao imageDao;
    private MarkovChainResultDao resultDao;

    @BeforeEach
    public void setup() throws SQLException {
        // Clean up previous run
        File dbFile = new File(TEST_DB);
        if (dbFile.exists()) {
            dbFile.delete();
        }

        SqliteInitializer.initialize(TEST_DB);
        imageDao = new DigitImageDao(TEST_DB);
        resultDao = new MarkovChainResultDao(TEST_DB);
    }

    @AfterEach
    public void teardown() {
        File dbFile = new File(TEST_DB);
        if (dbFile.exists()) {
            dbFile.delete();
        }
        File walFile = new File(TEST_DB + "-wal");
        if (walFile.exists()) {
            walFile.delete();
        }
        File shmFile = new File(TEST_DB + "-shm");
        if (shmFile.exists()) {
            shmFile.delete();
        }
    }

    @Test
    public void testDigitImageCrud() throws SQLException {
        DigitImage img1 = imageDao.getOrCreateByPath("train/5/123.png", "hash1");
        Assertions.assertNotNull(img1);
        Assertions.assertEquals("train/5/123.png", img1.getImageRelPath());
        Assertions.assertEquals("hash1", img1.getImageHash());

        // Retreive existing
        DigitImage img2 = imageDao.getOrCreateByPath("train/5/123.png", "hash1");
        Assertions.assertEquals(img1.getId(), img2.getId());

        // Helper find
        Optional<DigitImage> found = imageDao.findByPath("train/5/123.png");
        Assertions.assertTrue(found.isPresent());
        Assertions.assertEquals(img1.getId(), found.get().getId());
    }

    @Test
    public void testMarkovChainResultCrud() throws SQLException {
        DigitImage img = imageDao.getOrCreateByPath("img1.png", "h1");
        double[] scores = { 0.1, 0.2, 0.3, 0.4, 0.5 };

        resultDao.upsertScores(img.getId(), "row", "v1", scores);

        Optional<double[]> loaded = resultDao.loadScores(img.getId(), "row", "v1");
        Assertions.assertTrue(loaded.isPresent());
        Assertions.assertArrayEquals(scores, loaded.get(), 0.0001);

        // Update
        double[] scores2 = { 0.9, 0.8, 0.7 };
        resultDao.upsertScores(img.getId(), "row", "v1", scores2);
        loaded = resultDao.loadScores(img.getId(), "row", "v1");
        Assertions.assertArrayEquals(scores2, loaded.get(), 0.0001);

        // Delete
        resultDao.deleteByChain("row", "v1");
        loaded = resultDao.loadScores(img.getId(), "row", "v1");
        Assertions.assertFalse(loaded.isPresent());
    }
}
