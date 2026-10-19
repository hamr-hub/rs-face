/* rs-face Platform · 可选 API Token 支持
 * ----------------------------------------------------------------------------
 * 后端只有在设置了 RSFACE_API_TOKEN 时才校验 Bearer Token；未设置时本模块
 * 完全是无害透传(localStorage 里没有 token 就不加任何头)。
 *
 * 设计为「零侵入」:脚本必须在其它业务脚本之前加载,然后:
 *   1. 包装 window.fetch —— 对同源 /api 请求自动附带
 *      `Authorization: Bearer <token>`;
 *   2. 包装 window.EventSource —— SSE 无法自定义请求头,改用
 *      `?access_token=<token>` 查询参数(后端同权识别);
 *   3. 任意 /api 请求收到 401 时弹一次录入框,保存到 localStorage 后重试
 *      当前页面;连续失败可清除 token。
 *
 * 静态资源 / /media/* 不附带 token(后端也不校验),避免把凭据泄露到
 * <img>/<video> 的 URL 里。
 */
'use strict';

const RsAuth = (() => {
  const STORAGE_KEY = 'rsface.apiToken';
  let prompting = false; // 防止多个并发 401 弹多个框

  const getToken = () => {
    try {
      return localStorage.getItem(STORAGE_KEY) || '';
    } catch {
      return '';
    }
  };

  const setToken = (t) => {
    try {
      if (t) localStorage.setItem(STORAGE_KEY, t);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* 隐私模式等场景:静默忽略,本次会话仍可用内存值 */
    }
  };

  /** 仅对同源相对路径的 /api 请求附加凭据;绝对 URL(外域/媒体)不动。 */
  const isProtectedApiUrl = (input) => {
    let url = input;
    if (typeof input === 'object' && input && 'url' in input) url = input.url;
    if (typeof url !== 'string') return false;
    if (/^https?:\/\//i.test(url)) {
      try {
        if (new URL(url).origin !== window.location.origin) return false;
      } catch {
        return false;
      }
    }
    return url.includes('/api/');
  };

  /** 给 Request 的 headers 追加 Authorization,不破坏 FormData 的 boundary。 */
  const withAuth = (input, init, token) => {
    // 走 Request 对象形式:复制其 headers 后追加,body/method 原样保留。
    if (typeof input === 'object' && input instanceof Request) {
      const headers = new Headers(input.headers);
      headers.set('Authorization', `Bearer ${token}`);
      return new Request(input, { headers, ...(init || {}) });
    }
    const headers = new Headers((init && init.headers) || undefined);
    headers.set('Authorization', `Bearer ${token}`);
    return [input, { ...(init || {}), headers }];
  };

  /** 401 时收集 token;返回 true 表示用户已更新 token、调用方可刷新重试。 */
  const promptForToken = async (resp) => {
    if (prompting) return false;
    prompting = true;
    try {
      const current = getToken();
      const hint = current
        ? 'Token 无效或已过期,请重新输入:'
        : '服务端开启了 API Token 鉴权,请输入访问 Token:';
      const next = window.prompt(hint, current || '');
      if (next === null) {
        // 用户取消:如果本来就 401,清掉旧 token 并提示。
        if (current) {
          setToken('');
          window.alert('未提供有效 Token,该请求被拒绝(401)。');
        }
        return false;
      }
      const trimmed = next.trim();
      if (!trimmed) return false;
      setToken(trimmed);
      // 简单粗暴:拿到新 token 后整页重载,让所有轮询/SSE 用新凭据重建。
      window.location.reload();
      return true;
    } finally {
      // reload 后本脚本会重新执行;未 reload 时释放标志位。
      setTimeout(() => {
        prompting = false;
      }, 0);
    }
  };

  const install = () => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const token = getToken();
      let callInput = input;
      let callInit = init;
      if (token && isProtectedApiUrl(input)) {
        const patched = withAuth(input, init, token);
        if (patched instanceof Request) {
          callInput = patched;
          callInit = undefined;
        } else {
          [callInput, callInit] = patched;
        }
      }
      return originalFetch(callInput, callInit).then((resp) => {
        if (resp.status === 401 && isProtectedApiUrl(input)) {
          // 后台静默触发录入框(不阻塞这个 promise;reload 会接管页面)。
          promptForToken(resp);
        }
        return resp;
      });
    };

    // SSE:EventSource 不能带自定义头,用查询参数兜底。
    const OriginalEventSource = window.EventSource;
    if (OriginalEventSource) {
      window.EventSource = function (url, config) {
        const token = getToken();
        if (token && typeof url === 'string' && url.includes('/api/')) {
          const sep = url.includes('?') ? '&' : '?';
          url = `${url}${sep}access_token=${encodeURIComponent(token)}`;
        }
        return new OriginalEventSource(url, config);
      };
      window.EventSource.prototype = OriginalEventSource.prototype;
    }
  };

  install();

  return { getToken, setToken, isProtectedApiUrl };
})();
