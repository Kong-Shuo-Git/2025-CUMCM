package com.sdmu.cumcm2025.model.dto;

import lombok.Data;

/**
 * 测量请求参数DTO
 */
@Data
public class MeasurementRequest {
    private double incidentAngle = 10.0;     // 入射角(度)
    private double thickness = 7.32;        // 外延层厚度(μm)
    private double nAir = 1.0;              // 空气折射率
    private double nSic = 2.52;             // 碳化硅外延层折射率
    private double nSubstrate = 3.05;       // 衬底折射率
    private int numBeams = 5;               // 显示的光束数量
}