package com.markovai.db;

import com.markovai.util.DoubleArrayCodec;

import java.sql.*;
import java.util.Optional;

public class MarkovChainResultDao {

    private final String dbPath;

    public MarkovChainResultDao(String dbPath) {
        this.dbPath = dbPath;
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection("jdbc:sqlite:" + dbPath);
    }

    public Optional<double[]> loadScores(long imageId, String chainType, String chainVersion) throws SQLException {
        String sql = "SELECT scores_blob FROM markov_chain_result " +
                "WHERE image_id = ? AND chain_type = ? AND chain_version = ?";
        try (Connection conn = connect();
                PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, imageId);
            ps.setString(2, chainType);
            ps.setString(3, chainVersion);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    byte[] blob = rs.getBytes("scores_blob");
                    return Optional.ofNullable(DoubleArrayCodec.fromBytes(blob));
                }
            }
        }
        return Optional.empty();
    }

    public void upsertScores(long imageId, String chainType, String chainVersion, double[] scores) throws SQLException {
        byte[] blob = DoubleArrayCodec.toBytes(scores);
        long now = System.currentTimeMillis();
        String sql = "INSERT INTO markov_chain_result (image_id, chain_type, chain_version, scores_blob, created_ts) " +
                "VALUES (?, ?, ?, ?, ?) " +
                "ON CONFLICT(image_id, chain_type, chain_version) DO UPDATE SET " +
                "scores_blob = excluded.scores_blob, created_ts = excluded.created_ts";

        try (Connection conn = connect();
                PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, imageId);
            ps.setString(2, chainType);
            ps.setString(3, chainVersion);
            ps.setBytes(4, blob);
            ps.setLong(5, now);
            ps.executeUpdate();
        }
    }

    public void deleteByChain(String chainType, String chainVersion) throws SQLException {
        String sql = "DELETE FROM markov_chain_result WHERE chain_type = ? AND chain_version = ?";
        try (Connection conn = connect();
                PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, chainType);
            ps.setString(2, chainVersion);
            ps.executeUpdate();
        }
    }

    public void deleteByImage(long imageId) throws SQLException {
        String sql = "DELETE FROM markov_chain_result WHERE image_id = ?";
        try (Connection conn = connect();
                PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, imageId);
            ps.executeUpdate();
        }
    }
}
