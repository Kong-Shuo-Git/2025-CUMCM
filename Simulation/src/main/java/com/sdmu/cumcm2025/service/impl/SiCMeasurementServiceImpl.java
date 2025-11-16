package com.sdmu.cumcm2025.service.impl;

import com.sdmu.cumcm2025.model.dto.MeasurementRequest;
import com.sdmu.cumcm2025.model.dto.MeasurementResult;
import com.sdmu.cumcm2025.service.SiCMeasurementService;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 碳化硅外延层测量服务实现类
 */
@Service
public class SiCMeasurementServiceImpl implements SiCMeasurementService {

    private final Random random = new Random();

    @Override
    public MeasurementResult performMeasurement(MeasurementRequest request) {
        MeasurementResult result = new MeasurementResult();
        
        // 设置测量参数
        result.setIncidentAngle(request.getIncidentAngle());
        result.setNSic(request.getNSic());
        result.setNSubstrate(request.getNSubstrate());
        result.setSetThickness(request.getThickness());
        result.setNumBeams(request.getNumBeams());
        
        // 计算反射率
        double R1 = Math.pow(calculateReflectionCoefficient(request.getNAir(), request.getNSic(), request.getIncidentAngle()), 2);
        double R2 = Math.pow(calculateReflectionCoefficient(request.getNSic(), request.getNSubstrate(), request.getIncidentAngle()), 2);
        result.setR1(R1);
        result.setR2(R2);
        result.setStrongInterference(R1 > 0.1 && R2 > 0.1);
        
        // 计算厚度
        double calculatedThickness = calculateThicknessFromInterference(request);
        result.setCalculatedThickness(calculatedThickness);
        result.setRelativeError(Math.abs(calculatedThickness - request.getThickness()) / request.getThickness() * 100);
        
        // 生成干涉光谱数据
        result.setInterferenceData(generateInterferenceData(request));
        
        // 生成反射率光谱数据
        result.setSpectrumData(generateSpectrumData(request));
        
        // 生成多角度测量数据
        result.setMultiAngleData(generateMultiAngleData(request));
        
        return result;
    }

    @Override
    public double calculateReflectionCoefficient(double n1, double n2, double angleDeg) {
        double thetaI = Math.toRadians(angleDeg);
        try {
            // 使用 Snell 定律求折射角
            double sinThetaT = (n1 / n2) * Math.sin(thetaI);
            if (Math.abs(sinThetaT) >= 1.0) {
                return 1.0; // 全反射
            }
            double thetaT = Math.asin(sinThetaT);
            // s-偏振反射系数
            double r = (n1 * Math.cos(thetaI) - n2 * Math.cos(thetaT)) / 
                    (n1 * Math.cos(thetaI) + n2 * Math.cos(thetaT));
            return Math.abs(r);
        } catch (Exception e) {
            return 1.0;
        }
    }

    @Override
    public double calculatePhaseDifference(double wavelength, double thickness, double n, double angleDeg) {
        double thetaI = Math.toRadians(angleDeg);
        try {
            double sinThetaT = (1.0 / n) * Math.sin(thetaI); // n_air = 1
            if (Math.abs(sinThetaT) >= 1.0) {
                return 0.0;
            }
            double cosThetaT = Math.sqrt(1 - sinThetaT * sinThetaT);
            double phase = 4 * Math.PI * n * thickness * cosThetaT / wavelength;
            return phase;
        } catch (Exception e) {
            return 0.0;
        }
    }

    @Override
    public double airyReflectance(double R1, double R2, double phase) {
        double sqrtR = Math.sqrt(R1 * R2);
        double numerator = R1 + R2 + 2 * sqrtR * Math.cos(phase);
        double denominator = 1 + R1 * R2 + 2 * sqrtR * Math.cos(phase);
        return denominator != 0 ? numerator / denominator : 0;
    }

    @Override
    public double calculateThicknessFromInterference(MeasurementRequest request) {
        // 生成波长范围
        int numPoints = 800;
        double minWavelength = 3.0;
        double maxWavelength = 8.0;
        double[] wavelengths = new double[numPoints];
        double[] reflectances = new double[numPoints];
        
        for (int i = 0; i < numPoints; i++) {
            wavelengths[i] = minWavelength + (maxWavelength - minWavelength) * i / (numPoints - 1);
        }
        
        // 计算反射率
        double R1 = Math.pow(calculateReflectionCoefficient(request.getNAir(), request.getNSic(), request.getIncidentAngle()), 2);
        double R2 = Math.pow(calculateReflectionCoefficient(request.getNSic(), request.getNSubstrate(), request.getIncidentAngle()), 2);
        
        for (int i = 0; i < numPoints; i++) {
            double phase = calculatePhaseDifference(wavelengths[i], request.getThickness(), request.getNSic(), request.getIncidentAngle());
            reflectances[i] = airyReflectance(R1, R2, phase);
        }
        
        // 找极大值
        List<Integer> maximaIndices = new ArrayList<>();
        for (int i = 15; i < numPoints - 15; i++) {
            boolean isMax = true;
            for (int j = i - 15; j <= i + 15; j++) {
                if (j != i && reflectances[j] > reflectances[i]) {
                    isMax = false;
                    break;
                }
            }
            if (isMax) {
                maximaIndices.add(i);
            }
        }
        
        if (maximaIndices.size() < 2) {
            return request.getThickness();
        }
        
        // 取中间两相邻峰
        int mid = maximaIndices.size() / 2;
        int idx1 = maximaIndices.get(mid);
        int idx2 = maximaIndices.get(mid + 1);
        double lambda1 = wavelengths[idx1];
        double lambda2 = wavelengths[idx2];
        
        // 转换为波数差
        double deltaSigma = Math.abs(1/lambda1 - 1/lambda2) * 1e4; // cm⁻¹
        
        // 有效折射率修正入射角
        double cosTerm = Math.sqrt(request.getNSic() * request.getNSic() - Math.sin(Math.toRadians(request.getIncidentAngle())) * Math.sin(Math.toRadians(request.getIncidentAngle())));
        double calculatedD = 1 / (2 * deltaSigma * cosTerm); // 单位：cm
        
        return calculatedD * 1e4; // cm → μm
    }

    @Override
    public List<double[]> generateInterferenceData(MeasurementRequest request) {
        List<double[]> data = new ArrayList<>();
        int numPoints = 500;
        double minWavelength = 2.0;
        double maxWavelength = 10.0;
        
        double R1 = Math.pow(calculateReflectionCoefficient(request.getNAir(), request.getNSic(), request.getIncidentAngle()), 2);
        double R2 = Math.pow(calculateReflectionCoefficient(request.getNSic(), request.getNSubstrate(), request.getIncidentAngle()), 2);
        
        for (int i = 0; i < numPoints; i++) {
            double wavelength = minWavelength + (maxWavelength - minWavelength) * i / (numPoints - 1);
            double phase = calculatePhaseDifference(wavelength, request.getThickness(), request.getNSic(), request.getIncidentAngle());
            double reflectance = airyReflectance(R1, R2, phase);
            data.add(new double[]{wavelength, reflectance});
        }
        
        return data;
    }

    @Override
    public List<double[]> generateSpectrumData(MeasurementRequest request) {
        List<double[]> data = new ArrayList<>();
        int numPoints = 500;
        double minWavenumber = 1000;
        double maxWavenumber = 5000;
        
        double R1 = Math.pow(calculateReflectionCoefficient(request.getNAir(), request.getNSic(), request.getIncidentAngle()), 2);
        double R2 = Math.pow(calculateReflectionCoefficient(request.getNSic(), request.getNSubstrate(), request.getIncidentAngle()), 2);
        
        for (int i = 0; i < numPoints; i++) {
            double wavenumber = minWavenumber + (maxWavenumber - minWavenumber) * i / (numPoints - 1);
            double wavelength = 1e4 / wavenumber; // μm
            double phase = calculatePhaseDifference(wavelength, request.getThickness(), request.getNSic(), request.getIncidentAngle());
            double theoretical = airyReflectance(R1, R2, phase);
            // 添加噪声模拟实验数据
            double noise = random.nextGaussian() * 0.015;
            double experimental = Math.max(0, Math.min(1, theoretical + noise));
            data.add(new double[]{wavenumber, experimental});
        }
        
        return data;
    }

    @Override
    public List<double[]> generateMultiAngleData(MeasurementRequest request) {
        List<double[]> data = new ArrayList<>();
        double originalAngle = request.getIncidentAngle();
        
        for (double angle = 5; angle <= 30; angle += 5) {
            request.setIncidentAngle(angle);
            double thickness = calculateThicknessFromInterference(request);
            // 添加噪声
            thickness *= (1 + random.nextGaussian() * 0.015);
            data.add(new double[]{angle, thickness});
        }
        
        // 恢复原始角度
        request.setIncidentAngle(originalAngle);
        
        return data;
    }
}