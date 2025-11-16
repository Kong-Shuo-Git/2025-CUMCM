package com.sdmu.cumcm2025.controller;

import com.sdmu.cumcm2025.model.dto.MeasurementRequest;
import com.sdmu.cumcm2025.model.dto.MeasurementResult;
import com.sdmu.cumcm2025.service.SiCMeasurementService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;

/**
 * 碳化硅外延层测量控制器
 */
@Controller
public class SiCMeasurementController {

    @Autowired
    private SiCMeasurementService measurementService;

    /**
     * 首页入口
     */
    @GetMapping("/")
    public String index() {
        return "index";
    }

    /**
     * 执行厚度测量的API接口
     */
    @PostMapping("/api/measure")
    @ResponseBody
    public MeasurementResult performMeasurement(@RequestBody MeasurementRequest request) {
        return measurementService.performMeasurement(request);
    }

    /**
     * 获取默认参数
     */
    @GetMapping("/api/defaults")
    @ResponseBody
    public MeasurementRequest getDefaultParameters() {
        return new MeasurementRequest();
    }
}