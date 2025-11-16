package com.sdmu.cumcm2025.service;

import com.sdmu.cumcm2025.model.dto.MeasurementRequest;
import com.sdmu.cumcm2025.model.dto.MeasurementResult;

import java.util.List;

/**
 * 碳化硅外延层测量服务接口
 */
public interface SiCMeasurementService {

    /**
     * 执行厚度测量计算
     * @param request 测量请求参数
     * @return 测量结果
     */
    MeasurementResult performMeasurement(MeasurementRequest request);

    /**
     * 计算反射系数
     * @param n1 第一介质折射率
     * @param n2 第二介质折射率
     * @param incidentAngle 入射角
     * @return 反射系数
     */
    double calculateReflectionCoefficient(double n1, double n2, double incidentAngle);

    /**
     * 计算相位差
     * @param wavelength 波长
     * @param thickness 厚度
     * @param n 折射率
     * @param incidentAngle 入射角
     * @return 相位差
     */
    double calculatePhaseDifference(double wavelength, double thickness, double n, double incidentAngle);

    /**
     * 计算艾里反射率
     * @param R1 第一界面反射率
     * @param R2 第二界面反射率
     * @param phase 相位差
     * @return 总反射率
     */
    double airyReflectance(double R1, double R2, double phase);

    /**
     * 根据干涉计算厚度
     * @param request 测量请求参数
     * @return 计算得到的厚度
     */
    double calculateThicknessFromInterference(MeasurementRequest request);

    /**
     * 生成干涉光谱数据
     * @param request 测量请求参数
     * @return 干涉光谱数据列表
     */
    List<double[]> generateInterferenceData(MeasurementRequest request);

    /**
     * 生成反射率光谱数据
     * @param request 测量请求参数
     * @return 反射率光谱数据列表
     */
    List<double[]> generateSpectrumData(MeasurementRequest request);

    /**
     * 生成多角度测量数据
     * @param request 测量请求参数
     * @return 多角度测量数据列表
     */
    List<double[]> generateMultiAngleData(MeasurementRequest request);
}