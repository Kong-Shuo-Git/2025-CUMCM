package com.sdmu.cumcm2025.model.dto;

import lombok.Data;

import java.util.List;

/**
 * 测量结果响应DTO
 */
@Data
public class MeasurementResult {
    // 测量参数
    private double incidentAngle;
    private double nSic;
    private double nSubstrate;
    private double setThickness;
    
    // 计算结果
    private double calculatedThickness;
    private double relativeError;
    private double r1;
    private double r2;
    private boolean strongInterference;
    private int numBeams;
    
    // 光谱数据
    private List<double[]> interferenceData; // 干涉光谱数据 [wavelength, reflectance]
    private List<double[]> spectrumData;      // 反射率光谱数据 [wavenumber, reflectance]
    private List<double[]> multiAngleData;    // 多角度测量数据 [angle, thickness]
}