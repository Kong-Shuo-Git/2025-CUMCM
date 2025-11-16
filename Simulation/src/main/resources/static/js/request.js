// 响应拦截器
axios.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response && error.response.status === 401) {
            // 使用原生alert替代Element UI
            alert("401 权限不够跳转到登录页面");
            setTimeout(()=>{
                window.location.href = '/page/login.html';
            },2000);

        }
        return Promise.reject(error);
    }
);