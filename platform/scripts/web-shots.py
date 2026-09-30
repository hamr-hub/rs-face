#!/usr/bin/env python3
"""Playwright screenshot + DOM 验收:截空态 / 任务列表 / 详情(已存在任务) / 主题切换。

更新于 2026-09-30 适配 DESIGN.md 重设计后的新 UI:
  - 顶栏 3 元素(logo+title / search / theme),无 #tb-new/#tb-settings/#tb-help
  - 侧栏 1 列,3 段时间分组(进行 / 今天 / 较早),无 .kpi / .sb-filter / .new-type
  - 主区三态 empty/detail,无 #tb-dashboard / Inspector 抽屉
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE", "http://localhost:20080")
OUT = os.environ.get("OUT_DIR", "/tmp/rsface-shots")
os.makedirs(OUT, exist_ok=True)


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/snap/bin/chromium",
            args=["--no-sandbox", "--disable-gpu"],
            headless=True,
        )
        ctx = browser.new_context(viewport={"width": 1600, "height": 900}, color_scheme="dark")
        page = ctx.new_page()

        errors = []
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e.message}"))

        # 1) 空态(若有任务则不是 empty-state)
        page.goto(BASE + "/", wait_until="networkidle")
        # 等待 jobs 真正加载完成(可能比 networkidle 慢,因为还有第二波请求)
        try:
            page.wait_for_function(
                "() => (document.getElementById('sb-count')?.textContent || '0') !== '0' || document.querySelectorAll('.sb-item').length > 0",
                timeout=8000,
            )
        except Exception:
            print("[warn] jobs didn't load within 8s, proceeding anyway")
        page.wait_for_timeout(400)
        page.screenshot(path=f"{OUT}/01-home-dark.png")

        info = page.evaluate("""() => ({
            theme: document.documentElement.getAttribute('data-theme'),
            bodyBg: getComputedStyle(document.body).backgroundColor,
            topbar: [...document.querySelectorAll('#topbar > *')].map(e => e.tagName + '#' + e.id).filter(x => !x.endsWith('#')),
            sidebarVisible: !document.getElementById('sidebar').classList.contains('hidden'),
            previewVisible: !document.getElementById('preview').classList.contains('hidden'),
            emptyVisible: !document.getElementById('pv-empty')?.classList.contains('hidden'),
            detailVisible: !document.getElementById('pv-detail')?.classList.contains('hidden'),
            sbCount: document.getElementById('sb-count')?.textContent,
            sbFoot: document.getElementById('sb-foot')?.textContent,
            sbTotal: document.getElementById('sb-total')?.textContent,
            statusText: document.getElementById('status-bar')?.textContent?.trim().slice(0, 80),
            jobsLoaded: document.querySelectorAll('.sb-item').length,
        })""")
        print("[home dark]", info)

        # 2) 切到 light 主题(点击 tb-theme 两次:auto -> dark -> light)
        page.click("#tb-theme"); page.wait_for_timeout(120)
        page.click("#tb-theme"); page.wait_for_timeout(120)
        page.screenshot(path=f"{OUT}/02-home-light.png")
        light_info = page.evaluate("() => ({ theme: document.documentElement.getAttribute('data-theme') })")
        print("[home light]", light_info)
        # 切回 dark 给后续截图一致背景
        page.click("#tb-theme"); page.wait_for_timeout(120)

        # 3) 打开第一个任务(如有)
        first_job = page.evaluate("""() => {
            const it = document.querySelector('.sb-item');
            return it ? it.id.replace(/^sb-j-/, '') : null;
        }""")
        if first_job:
            # 虚拟滚动会重复挂载/卸载,直接派发 click 事件更稳定
            page.evaluate(
                "(id) => document.getElementById('sb-j-' + id)?.click()",
                first_job,
            )
            page.wait_for_function(
                "() => !document.getElementById('pv-detail').classList.contains('hidden')",
                timeout=8000,
            )
            page.wait_for_timeout(400)
            page.screenshot(path=f"{OUT}/03-detail-dark.png")
            detail_info = page.evaluate("""() => ({
                title: document.getElementById('pv-title')?.textContent,
                status: document.getElementById('pv-status')?.textContent,
                faces: document.getElementById('pv-faces')?.textContent,
                gpu: document.getElementById('pv-gpu')?.textContent,
                overlaySize: document.getElementById('pv-overlay') ? {
                    w: document.getElementById('pv-overlay').width,
                    h: document.getElementById('pv-overlay').height,
                } : null,
                faceCardCount: document.querySelectorAll('.face-card').length,
                overlayCanvas: !!document.querySelector('#pv-overlay'),
                imgLoaded: document.getElementById('pv-img')?.classList.contains('is-loaded'),
            })""")
            print(f"[detail {first_job[:8]}]", detail_info)

            # 4) 切到原图(按数字键 1)再截一张
            page.keyboard.press("1"); page.wait_for_timeout(400)
            page.screenshot(path=f"{OUT}/04-detail-original.png")

            # 5) kebab 菜单
            page.click("#pv-kebab"); page.wait_for_timeout(300)
            page.screenshot(path=f"{OUT}/05-detail-kebab.png")
            page.keyboard.press("Escape")

        # 6) "新建" 弹窗(空态按钮)
        page.evaluate("preview && preview.close && preview.close()")  # 切回空态
        page.wait_for_timeout(200)
        new_btn = page.query_selector("#pv-empty-new")
        if new_btn:
            new_btn.click()
            page.wait_for_timeout(300)
            page.screenshot(path=f"{OUT}/06-modal-new-dark.png")
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)

        # 7) 主题切换到 light 后再看详情(如有)
        page.click("#tb-theme"); page.wait_for_timeout(120)
        page.click("#tb-theme"); page.wait_for_timeout(120)
        if first_job:
            page.evaluate(
                "(id) => document.getElementById('sb-j-' + id)?.click()",
                first_job,
            )
            page.wait_for_function(
                "() => !document.getElementById('pv-detail').classList.contains('hidden')",
                timeout=8000,
            )
            page.wait_for_timeout(400)
            page.screenshot(path=f"{OUT}/07-detail-light.png")
            page.keyboard.press("Escape")

        # 8) 顶栏元素可访问性核验
        a11y = page.evaluate("""() => {
            const ids = ['tb-drawer', 'tb-home', 'tb-search', 'tb-search-clear', 'tb-theme'];
            return ids.map(id => ({ id, hasEl: !!document.getElementById(id), display: document.getElementById(id) && getComputedStyle(document.getElementById(id)).display }));
        }""")
        print("[topbar a11y]", a11y)

        # 9) 全局拖入 overlay 是否存在(隐藏态)
        dz_overlay = page.evaluate("() => ({ exists: !!document.getElementById('dropzone-overlay'), hidden: document.getElementById('dropzone-overlay')?.classList.contains('hidden') })")
        print("[dropzone overlay]", dz_overlay)

        print("\nConsole errors:", len(errors), errors[:5] if errors else "(none)")
        browser.close()


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)