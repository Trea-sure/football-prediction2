# ==================== 足球智能预测系统 v5.2 ====================
# 优化：直接从HTML隐藏input提取赔率数据，修复比分提取，保持原始顺序

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 深度学习
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 爬虫
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# 可视化
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== 配置 ====================

@dataclass
class Config:
    DATA_FILE: str = "football_training_data.json"
    MODEL_DIR: str = "models_v5"
    MIN_TRAIN_SAMPLES: int = 30
    
    def __post_init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)

CONFIG = Config()

# ==================== 数据持久化管理 ====================

class DataPersistence:
    """数据持久化管理器 - 自动保存到本地"""

    DATA_FILE = "football_data_cache.json"

    def __init__(self):
        self.data = self._load_data()

    def _load_data(self):
        """从本地加载数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 从本地加载了 {len(data)} 场比赛数据")

                # 转换为DataFrame
                df = pd.DataFrame(data)

                # 解析JSON字符串为列表
                for col in ['europe', 'asia', 'daxiao', 'handicap']:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: json.loads(x) if isinstance(x, str) and x.strip() else 
                                     (x if isinstance(x, list) else [])
                        )

                # 确保必要列存在
                required_cols = ['match_id', 'date', 'league', 'time', 'home_team', 'away_team', 'actual_result']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = ''

                return df
            except Exception as e:
                print(f"⚠️ 加载本地数据失败: {e}")

        # 返回空DataFrame
        return pd.DataFrame(columns=[
            'match_id', 'date', 'league', 'time', 'status', 'home_team', 'away_team',
            'score', 'score_home', 'score_away', 'actual_result', 'has_result',
            'europe', 'asia', 'daxiao', 'handicap', 'order'
        ])

    def save_data(self):
        """保存数据到本地"""
        try:
            with open(self.DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False

    def add_matches(self, matches):
        """批量添加比赛"""
        added, updated = 0, 0
        for match in matches:
            match_id = match.get('match_id')
            existing_idx = None
            for idx, m in enumerate(self.data):
                if m.get('match_id') == match_id:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                existing = self.data[existing_idx]
                for key in ['europe', 'asia', 'handicap', 'daxiao']:
                    if key in match and match[key]:
                        existing[key] = match[key]
                for key, value in match.items():
                    if key not in ['europe', 'asia', 'handicap', 'daxiao']:
                        existing[key] = value
                self.data[existing_idx] = existing
                updated += 1
            else:
                self.data.append(match)
                added += 1

        self.save_data()
        return added, updated

    def get_trainable_matches(self):
        """获取可用于训练的比赛"""
        result = []
        for m in self.data:
            has_result = m.get('actual_result') in ['主胜', '平局', '客胜']
            has_odds = any(len(m.get(ot, []) or []) > 0 for ot in ['europe', 'asia', 'handicap', 'daxiao'])
            if has_result and has_odds:
                result.append(m)
        return result

    def get_statistics(self):
        """获取统计"""
        trainable = self.get_trainable_matches()
        result_dist = {"主胜": 0, "平局": 0, "客胜": 0}
        for m in trainable:
            r = m.get('actual_result')
            if r in result_dist:
                result_dist[r] += 1
        return {
            'total': len(self.data),
            'trainable': len(trainable),
            'result_distribution': result_dist
        }


# ==================== 数据采集模块 ====================

class DataCollector:
    """修复版数据采集器，适配500.com实际HTML结构"""

    def __init__(self):
        self.driver = None
        self.base_urls = {
            'live': "https://live.500.com/",
            'europe': "https://odds.500.com/fenxi/ouzhi-{}.shtml",
            'handicap': "https://odds.500.com/fenxi/rangqiu-{}.shtml",
            'asia': "https://odds.500.com/fenxi/yazhi-{}.shtml",
            'daxiao': "https://odds.500.com/fenxi/daxiao-{}.shtml"
        }
        self.log_callback = None

    def set_log_callback(self, callback):
        self.log_callback = callback

    def _log(self, message):
        if self.log_callback:
            self.log_callback(message)
        print(message)

    def get_driver(self):
        if self.driver is not None:
            try:
                self.driver.current_url
                return self.driver
            except:
                self.close()

        try:
            options = Options()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-gpu")
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})

            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })
            self.driver.set_page_load_timeout(30)
            return self.driver
        except Exception as e:
            st.error(f"浏览器创建失败: {e}")
            return None

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None

    def get_page(self, url, wait=3):
        driver = self.get_driver()
        if not driver:
            return None

        try:
            driver.get(url)
            time.sleep(wait)
            for _ in range(2):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)
            return driver.page_source
        except Exception as e:
            return None

    def fetch_matches_by_date(self, date_str: str, only_finished: bool = True) -> pd.DataFrame:
        """【核心修复】获取指定日期的比赛数据"""
        html = self.get_page(f"{self.base_urls['live']}?e={date_str}", wait=4)
        if not html:
            return pd.DataFrame()

        soup = BeautifulSoup(html, 'lxml')
        matches = []

        # 【修复】使用 id="a数字" 格式匹配
        for idx, tr in enumerate(soup.find_all('tr', id=re.compile(r'^a\d+$'))):
            try:
                # 【修复】提取match_id（去掉'a'前缀）
                match_id_full = tr.get('id', '')
                match_id = match_id_full.replace('a', '') if match_id_full else ''
                if not match_id or not match_id.isdigit():
                    continue

                tds = tr.find_all('td')
                if len(tds) < 8:
                    continue

                # 【修复】状态判断（结合status属性和文本）
                status_code = tr.get('status', '')
                row_text = tr.get_text()

                status = "未"
                if status_code == '4' or '完' in row_text:
                    status = "完"
                elif status_code == '2' or '进行中' in row_text:
                    status = "进行中"

                if only_finished and status != "完":
                    continue

                # 提取联赛
                league = tds[1].get_text(strip=True) if len(tds) > 1 else ""

                # 提取时间
                match_time = ""
                if len(tds) > 3:
                    time_text = tds[3].get_text(strip=True)
                    time_match = re.search(r'(\d{2}:\d{2})', time_text)
                    if time_match:
                        match_time = time_match.group(1)

                # 【修复】提取比分 - 从pk div内的第一个和第三个<a>标签获取
                score_home, score_away, actual_result = "", "", ""
                if status == "完":
                    # 查找包含pk类的div
                    pk_div = tr.find("div", class_="pk")

                    if pk_div:
                        # 获取所有<a>标签，第一个是主队比分，第三个是客队比分
                        all_links = pk_div.find_all("a")
                        if len(all_links) >= 3:
                            try:
                                # 第一个<a>是主队比分
                                home_text = all_links[0].get_text(strip=True)
                                # 第三个<a>是客队比分
                                away_text = all_links[2].get_text(strip=True)

                                home_val = int(home_text)
                                away_val = int(away_text)
                                if 0 <= home_val <= 20 and 0 <= away_val <= 20:
                                    score_home = str(home_val)
                                    score_away = str(away_val)
                            except Exception as e:
                                # 静默失败，尝试备用方法
                                pass

                        # 如果上面的方法失败，尝试通过style颜色查找
                        if not score_home or not score_away:
                            try:
                                # 查找红色样式的<a>标签（通常是比分）
                                red_links = pk_div.find_all("a", style=lambda x: x and "red" in x.lower() if x else False)
                                if len(red_links) >= 2:
                                    home_val = int(red_links[0].get_text(strip=True))
                                    away_val = int(red_links[1].get_text(strip=True))
                                    if 0 <= home_val <= 20 and 0 <= away_val <= 20:
                                        score_home = str(home_val)
                                        score_away = str(away_val)
                            except:
                                pass

                    # 优先根据比分计算结果（最可靠）
                    if score_home and score_away:
                        try:
                            sh, sa = int(score_home), int(score_away)
                            if sh > sa:
                                actual_result = "主胜"
                            elif sh < sa:
                                actual_result = "客胜"
                            else:
                                actual_result = "平局"
                        except:
                            pass

                    # 如果比分计算失败，从HTML提取
                    if not actual_result:
                        red_tds = tr.find_all("td", class_="red")
                        for td in red_tds:
                            result_text = td.get_text(strip=True)
                            # 跳过比分格式（如"2 - 0"）和空文本
                            if not result_text or re.match(r'^\d+\s*[-:]\s*\d+$', result_text):
                                continue
                            # 匹配结果（从主队视角：胜=主胜，负=客胜，平=平局）
                            if result_text in ["主胜", "客胜", "平局"]:
                                actual_result = result_text
                                break
                            elif result_text == "平":
                                actual_result = "平局"
                                break
                            elif result_text in ["胜", "主", "主胜"]:
                                actual_result = "主胜"
                                break
                            elif result_text in ["负", "客", "客胜"]:
                                actual_result = "客胜"
                                break
                # 【修复】提取球队名称
                teams = []

                # 方法1: 从特定列提取
                for t_idx in [5, 7]:
                    if t_idx < len(tds):
                        links = tds[t_idx].find_all('a')
                        for link in links:
                            name = link.get_text(strip=True)
                            if name and len(name) > 1 and not re.match(r'^\d+(\.\d+)?$', name):
                                if name not in teams:
                                    teams.append(name)
                                    break

                # 方法2: 从所有文本提取
                if len(teams) < 2:
                    for td in tds[2:]:
                        text = td.get_text(strip=True)
                        if (2 <= len(text) <= 15 and 
                            not any(c.isdigit() for c in text) and
                            text not in ['主', '客', 'vs', '-', ':', '半球', '平手', '受半球', '受平手', '一球']):
                            if text not in teams:
                                teams.append(text)
                        if len(teams) >= 2:
                            break

                teams = teams[:2]

                if len(teams) >= 2:
                    matches.append({
                        'order': idx,
                        'match_id': match_id,
                        'date': date_str,
                        'league': league or '未知',
                        'time': match_time,
                        'status': status,
                        'home_team': teams[0],
                        'away_team': teams[1],
                        'score': f"{score_home}-{score_away}" if score_home and score_away else "",
                        'score_home': score_home,
                        'score_away': score_away,
                        'actual_result': actual_result,
                        'has_result': status == "完" and actual_result != ""
                    })

            except Exception as e:
                continue

        df = pd.DataFrame(matches)
        if not df.empty:
            df = df.sort_values('order').reset_index(drop=True)
        return df

    def fetch_odds_from_input(self, match_id: str, odds_type: str = 'europe') -> List[Dict]:
        """从HTML隐藏input中提取赔率数据"""
        url = self.base_urls[odds_type].format(match_id)
        self._log(f"📡 {url}")

        html = self.get_page(url, wait=2)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')
        companies = []

        try:
            # 查找隐藏的input元素，name="row"
            row_input = soup.find('input', {'name': 'row'})
            if row_input:
                row_value = row_input.get('value', '')
                if row_value:
                    company_rows = row_value.split('|')

                    for row in company_rows:
                        if not row or '公司' in row or '平均' in row:
                            continue

                        parts = row.split(',')
                        if len(parts) >= 7:
                            try:
                                company = {
                                    'company': parts[0].strip(),
                                    'init_home': float(parts[1]),
                                    'init_draw': float(parts[2]),
                                    'init_away': float(parts[3]),
                                    'live_home': float(parts[4]),
                                    'live_draw': float(parts[5]),
                                    'live_away': float(parts[6]),
                                }
                                company['change_home'] = round(company['live_home'] - company['init_home'], 2)
                                company['change_draw'] = round(company['live_draw'] - company['init_draw'], 2)
                                company['change_away'] = round(company['live_away'] - company['init_away'], 2)
                                companies.append(company)
                            except:
                                continue

            # 备用：从表格中提取
            if not companies:
                table = soup.find('table', id='datatb')
                if table:
                    for tr in table.find_all('tr'):
                        try:
                            tds = tr.find_all('td')
                            if len(tds) < 8:
                                continue

                            name = tds[0].get_text(strip=True)
                            if not name or any(x in name for x in ['平均', '最大', '公司']):
                                continue

                            def parse(txt):
                                try:
                                    return float(re.sub(r'[↑↓]', '', txt))
                                except:
                                    return 0.0

                            company = {
                                'company': name,
                                'init_home': parse(tds[2].get_text()),
                                'init_draw': parse(tds[3].get_text()),
                                'init_away': parse(tds[4].get_text()),
                                'live_home': parse(tds[5].get_text()),
                                'live_draw': parse(tds[6].get_text()),
                                'live_away': parse(tds[7].get_text()),
                            }
                            company['change_home'] = round(company['live_home'] - company['init_home'], 2)
                            company['change_draw'] = round(company['live_draw'] - company['init_draw'], 2)
                            company['change_away'] = round(company['live_away'] - company['init_away'], 2)

                            if company['live_home'] > 0 and company['live_draw'] > 0 and company['live_away'] > 0:
                                companies.append(company)
                        except:
                            continue

        except Exception as e:
            self._log(f"❌ 提取赔率失败: {str(e)[:50]}")

        return companies

    
    
    def fetch_all_odds(self, match_id: str, log_callback=None) -> Dict:
        """
        获取所有四种赔率 - 修复调用顺序
        """
        if log_callback:
            self.set_log_callback(log_callback)

        odds_data = {
            'europe': [],
            'asia': [],
            'handicap': [],
            'daxiao': []
        }

        self._log(f"🔄 开始获取比赛 {match_id} 的4种赔率...")

        # 1. 欧洲赔率（确保第一个执行）
        self._log(f"📊 [1/4] 开始获取欧洲赔率...")
        try:
            odds_data['europe'] = self._fetch_europe_odds(match_id)
            self._log(f"✓ [1/4] 欧洲赔率: {len(odds_data['europe'])} 家公司")
        except Exception as e:
            self._log(f"❌ [1/4] 欧洲赔率获取失败: {str(e)[:50]}")
            import traceback
            self._log(f"   错误详情: {traceback.format_exc()[:200]}")

        time.sleep(0.5)

        # 2. 亚盘
        self._log(f"📊 [2/4] 开始获取亚盘数据...")
        try:
            odds_data['asia'] = self._fetch_asia_odds(match_id)
            self._log(f"✓ [2/4] 亚盘: {len(odds_data['asia'])} 家公司")
        except Exception as e:
            self._log(f"❌ [2/4] 亚盘获取失败: {str(e)[:50]}")

        time.sleep(0.5)

        # 3. 让球胜平负
        self._log(f"📊 [3/4] 开始获取让球胜平负...")
        try:
            odds_data['handicap'] = self._fetch_handicap_odds(match_id)
            self._log(f"✓ [3/4] 让球: {len(odds_data['handicap'])} 家公司")
        except Exception as e:
            self._log(f"❌ [3/4] 让球获取失败: {str(e)[:50]}")

        time.sleep(0.5)

        # 4. 大小球
        self._log(f"📊 [4/4] 开始获取大小球数据...")
        try:
            odds_data['daxiao'] = self._fetch_daxiao_odds(match_id)
            self._log(f"✓ [4/4] 大小球: {len(odds_data['daxiao'])} 家公司")
        except Exception as e:
            self._log(f"❌ [4/4] 大小球获取失败: {str(e)[:50]}")

        total = sum(len(v) for v in odds_data.values())
        self._log(f"✅ 总计: {total} 条赔率数据 (欧赔:{len(odds_data['europe'])}, 亚盘:{len(odds_data['asia'])}, 让球:{len(odds_data['handicap'])}, 大小球:{len(odds_data['daxiao'])})")

        return odds_data

    
    
    
    
    def _fetch_europe_odds(self, match_id: str) -> List[Dict]:
        """
        获取欧洲赔率 - 根据实际HTML结构精确提取
        """
        url = f"https://odds.500.com/fenxi/ouzhi-{match_id}.shtml"
        self._log(f"      🌐 访问: {url}")

        html = self.get_page(url, wait=4)
        if not html:
            self._log(f"      ❌ 页面为空")
            return []

        soup = BeautifulSoup(html, 'lxml')
        companies = []

        # 查找所有数据行（根据实际HTML：tr[class="tr1"] 或 tr[class="tr2"]）
        rows = soup.find_all('tr', {'class': ['tr1', 'tr2']})
        self._log(f"      📋 找到 {len(rows)} 个数据行")

        for tr in rows:
            try:
                # 提取公司名
                company_name = ""
                name_td = tr.find('td', {'class': 'tb_plgs'})
                if name_td:
                    span = name_td.find('span', {'class': 'quancheng'})
                    if span:
                        company_name = span.get_text(strip=True)

                if not company_name or any(x in company_name for x in ['平均', '最大', '最小']):
                    continue

                # 查找包含赔率的嵌套表格
                # 根据HTML结构，赔率在第3个<td>中（索引2）
                tds = tr.find_all('td', recursive=False)
                if len(tds) < 3:
                    continue

                odds_td = tds[2]  # 第3个<td>包含赔率表格

                # 查找嵌套的赔率表格
                inner_table = odds_td.find('table', {'class': 'pl_table_data'})
                if not inner_table:
                    continue

                # 查找所有行
                inner_rows = inner_table.find_all('tr')
                if len(inner_rows) < 2:
                    continue

                # 第一行：初始赔率
                init_row = inner_rows[0]
                init_tds = init_row.find_all('td')
                if len(init_tds) < 3:
                    continue

                init_home = self._parse_odds_value(init_tds[0])
                init_draw = self._parse_odds_value(init_tds[1])
                init_away = self._parse_odds_value(init_tds[2])

                # 第二行：即时赔率
                live_row = inner_rows[1]
                live_tds = live_row.find_all('td')
                if len(live_tds) < 3:
                    continue

                live_home = self._parse_odds_value(live_tds[0])
                live_draw = self._parse_odds_value(live_tds[1])
                live_away = self._parse_odds_value(live_tds[2])

                # 验证数据
                if all(v > 0 for v in [init_home, init_draw, init_away, live_home, live_draw, live_away]):
                    companies.append({
                        'company': company_name,
                        'init_home': round(init_home, 2),
                        'init_draw': round(init_draw, 2),
                        'init_away': round(init_away, 2),
                        'live_home': round(live_home, 2),
                        'live_draw': round(live_draw, 2),
                        'live_away': round(live_away, 2),
                        'change_home': round(live_home - init_home, 2),
                        'change_draw': round(live_draw - init_draw, 2),
                        'change_away': round(live_away - init_away, 2)
                    })

                    # 显示第一个样本
                    if len(companies) == 1:
                        self._log(f"      ✓ 样本: {company_name} | 初:{init_home}/{init_draw}/{init_away} | 即:{live_home}/{live_draw}/{live_away}")

            except Exception as e:
                continue

        self._log(f"      ✅ 共获取 {len(companies)} 家公司")
        return companies

    def _parse_odds_value(self, td) -> float:
        """从td元素解析赔率值"""
        try:
            # 获取文本
            text = td.get_text(strip=True)
            # 清理
            text = re.sub(r'[↑↓]', '', text).strip()
            # 转换
            return float(text)
        except:
            return 0.0

    def _parse_odds_value(self, td) -> float:
        """从td元素解析赔率值"""
        try:
            text = td.get_text(strip=True)
            text = re.sub(r'[↑↓]', '', text).strip()
            return float(text)
        except:
            return 0.0

    def _fetch_asia_odds(self, match_id: str) -> List[Dict]:
        """
        获取亚盘数据
        数据格式: 公司名, 初始主水, 初始盘口, 初始客水, 即时主水, 即时盘口, 即时客水
        """
        url = f"https://odds.500.com/fenxi/yazhi-{match_id}.shtml"
        self._log(f"      🌐 访问: {url}")

        html = self.get_page(url, wait=3)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')
        companies = []

        try:
            # 从隐藏input获取
            row_input = soup.find('input', {'name': 'row'})
            if row_input:
                row_value = row_input.get('value', '')
                if row_value:
                    for row in row_value.split('|'):
                        if not row or '公司' in row:
                            continue
                        parts = row.split(',')
                        if len(parts) >= 7:
                            try:
                                companies.append({
                                    'company': parts[0].strip(),
                                    'init_home': float(parts[1]),      # 初始主水
                                    'init_handicap': parts[2].strip(),  # 初始盘口
                                    'init_away': float(parts[3]),       # 初始客水
                                    'live_home': float(parts[4]),       # 即时主水
                                    'live_handicap': parts[5].strip(),  # 即时盘口
                                    'live_away': float(parts[6])        # 即时客水
                                })
                            except:
                                continue

            # 备用：从表格获取
            if not companies:
                table = soup.find('table', id='datatb')
                if table:
                    for tr in table.find_all('tr')[1:]:
                        tds = tr.find_all('td')
                        if len(tds) >= 8:
                            try:
                                name = tds[0].get_text(strip=True)
                                if not name or any(x in name for x in ['平均', '最大']):
                                    continue

                                companies.append({
                                    'company': name,
                                    'init_home': self._parse_float(tds[2].get_text()),
                                    'init_handicap': tds[3].get_text(strip=True),
                                    'init_away': self._parse_float(tds[4].get_text()),
                                    'live_home': self._parse_float(tds[5].get_text()),
                                    'live_handicap': tds[6].get_text(strip=True),
                                    'live_away': self._parse_float(tds[7].get_text())
                                })
                            except:
                                continue
        except Exception as e:
            self._log(f"      ❌ 提取失败: {str(e)[:50]}")

        return companies

    def _fetch_handicap_odds(self, match_id: str) -> List[Dict]:
        """
        获取让球胜平负（竞彩）
        数据格式: 公司名, 让球数, 初始主胜, 初始平局, 初始客胜, 即时主胜, 即时平局, 即时客胜
        """
        url = f"https://odds.500.com/fenxi/rangqiu-{match_id}.shtml"
        self._log(f"      🌐 访问: {url}")

        html = self.get_page(url, wait=3)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')
        companies = []

        try:
            row_input = soup.find('input', {'name': 'row'})
            if row_input:
                row_value = row_input.get('value', '')
                if row_value:
                    for row in row_value.split('|'):
                        if not row or '公司' in row:
                            continue
                        parts = row.split(',')
                        if len(parts) >= 8:
                            try:
                                companies.append({
                                    'company': parts[0].strip(),
                                    'handicap': parts[1].strip(),       # 让球数
                                    'init_home': float(parts[2]),       # 初始让球主胜
                                    'init_draw': float(parts[3]),       # 初始让球平局
                                    'init_away': float(parts[4]),       # 初始让球客胜
                                    'live_home': float(parts[5]),       # 即时让球主胜
                                    'live_draw': float(parts[6]),       # 即时让球平局
                                    'live_away': float(parts[7])        # 即时让球客胜
                                })
                            except:
                                continue

            # 备用
            if not companies:
                table = soup.find('table', id='datatb')
                if table:
                    for tr in table.find_all('tr')[1:]:
                        tds = tr.find_all('td')
                        if len(tds) >= 9:
                            try:
                                name = tds[0].get_text(strip=True)
                                if not name or any(x in name for x in ['平均', '最大']):
                                    continue

                                companies.append({
                                    'company': name,
                                    'handicap': tds[1].get_text(strip=True),
                                    'init_home': self._parse_float(tds[3].get_text()),
                                    'init_draw': self._parse_float(tds[4].get_text()),
                                    'init_away': self._parse_float(tds[5].get_text()),
                                    'live_home': self._parse_float(tds[6].get_text()),
                                    'live_draw': self._parse_float(tds[7].get_text()),
                                    'live_away': self._parse_float(tds[8].get_text())
                                })
                            except:
                                continue
        except Exception as e:
            self._log(f"      ❌ 提取失败: {str(e)[:50]}")

        return companies

    def _fetch_daxiao_odds(self, match_id: str) -> List[Dict]:
        """
        获取大小球数据
        数据格式: 公司名, 初始大球水, 初始盘口, 初始小球水, 即时大球水, 即时盘口, 即时小球水
        """
        url = f"https://odds.500.com/fenxi/daxiao-{match_id}.shtml"
        self._log(f"      🌐 访问: {url}")

        html = self.get_page(url, wait=3)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')
        companies = []

        try:
            row_input = soup.find('input', {'name': 'row'})
            if row_input:
                row_value = row_input.get('value', '')
                if row_value:
                    for row in row_value.split('|'):
                        if not row or '公司' in row:
                            continue
                        parts = row.split(',')
                        if len(parts) >= 7:
                            try:
                                companies.append({
                                    'company': parts[0].strip(),
                                    'init_over': float(parts[1]),       # 初始大球水
                                    'init_line': parts[2].strip(),      # 初始盘口
                                    'init_under': float(parts[3]),      # 初始小球水
                                    'live_over': float(parts[4]),       # 即时大球水
                                    'live_line': parts[5].strip(),      # 即时盘口
                                    'live_under': float(parts[6])       # 即时小球水
                                })
                            except:
                                continue

            # 备用
            if not companies:
                table = soup.find('table', id='datatb')
                if table:
                    for tr in table.find_all('tr')[1:]:
                        tds = tr.find_all('td')
                        if len(tds) >= 8:
                            try:
                                name = tds[0].get_text(strip=True)
                                if not name or any(x in name for x in ['平均', '最大']):
                                    continue

                                companies.append({
                                    'company': name,
                                    'init_over': self._parse_float(tds[2].get_text()),
                                    'init_line': tds[3].get_text(strip=True),
                                    'init_under': self._parse_float(tds[4].get_text()),
                                    'live_over': self._parse_float(tds[5].get_text()),
                                    'live_line': tds[6].get_text(strip=True),
                                    'live_under': self._parse_float(tds[7].get_text())
                                })
                            except:
                                continue
        except Exception as e:
            self._log(f"      ❌ 提取失败: {str(e)[:50]}")

        return companies

    def _parse_float(self, text: str) -> float:
        """解析浮点数，处理特殊字符"""
        text = re.sub(r'[↑↓]', '', text).strip()
        try:
            return float(text)
        except:
            return 0.0

    def batch_fetch_history(self, start_date: str, days: int = 7, progress_callback=None, log_callback=None) -> pd.DataFrame:
        """批量获取历史数据"""
        if log_callback:
            self.set_log_callback(log_callback)

        all_matches = []
        start = datetime.strptime(start_date, '%Y-%m-%d')

        for i in range(days):
            current_date = start - timedelta(days=i)
            date_str = current_date.strftime('%Y-%m-%d')

            self._log(f"📅 获取 {date_str} 的比赛...")

            if progress_callback:
                progress_callback(i+1, days, date_str, "获取比赛列表...")

            matches = self.fetch_matches_by_date(date_str, only_finished=True)

            if not matches.empty:
                self._log(f"📝 找到 {len(matches)} 场完赛比赛")

                if progress_callback:
                    progress_callback(i+1, days, date_str, f"获取 {len(matches)} 场比赛赔率...")

                match_list = []
                for idx, row in matches.iterrows():
                    if progress_callback:
                        progress_callback(i+1, days, date_str, f"[{idx+1}/{len(matches)}] {row['home_team']} vs {row['away_team']}")

                    odds = self.fetch_all_odds(row['match_id'], log_callback)

                    match_data = row.to_dict()
                    match_data.update(odds)
                    match_list.append(match_data)

                    time.sleep(0.2)

                all_matches.extend(match_list)
                self._log(f"✅ {date_str} 完成，共 {len(match_list)} 场")
            else:
                self._log(f"⚠️ {date_str} 无完赛数据")

            time.sleep(0.5)

        df = pd.DataFrame(all_matches)
        if not df.empty:
            df = df.sort_values(['date', 'order']).reset_index(drop=True)
        return df


    def fetch_future_matches(self, date_str: str) -> pd.DataFrame:
        """获取指定日期的未来比赛（未开始或进行中）"""
        html = self.get_page(f"{self.base_urls['live']}?e={date_str}", wait=4)
        if not html:
            return pd.DataFrame()

        soup = BeautifulSoup(html, 'lxml')
        matches = []

        for idx, tr in enumerate(soup.find_all('tr', id=re.compile(r'^a\d+$'))):
            try:
                match_id_full = tr.get('id', '')
                match_id = match_id_full.replace('a', '') if match_id_full else ''
                if not match_id or not match_id.isdigit():
                    continue

                tds = tr.find_all('td')
                if len(tds) < 8:
                    continue

                status_code = tr.get('status', '')
                row_text = tr.get_text()

                status = "未"
                if status_code == '4' or '完' in row_text:
                    status = "完"
                elif status_code == '2' or '进行中' in row_text:
                    status = "进行中"

                # 只获取未开始或进行中的比赛
                if status == "完":
                    continue

                league = tds[1].get_text(strip=True) if len(tds) > 1 else ""

                match_time = ""
                if len(tds) > 3:
                    time_text = tds[3].get_text(strip=True)
                    time_match = re.search(r'(\d{2}:\d{2})', time_text)
                    if time_match:
                        match_time = time_match.group(1)

                # 提取球队名称
                teams = []
                for t_idx in [5, 7]:
                    if t_idx < len(tds):
                        links = tds[t_idx].find_all('a')
                        for link in links:
                            name = link.get_text(strip=True)
                            if name and len(name) > 1 and not re.match(r'^\d+(\.\d+)?$', name):
                                if name not in teams:
                                    teams.append(name)
                                    break

                if len(teams) < 2:
                    for td in tds[2:]:
                        text = td.get_text(strip=True)
                        if (2 <= len(text) <= 15 and 
                            not any(c.isdigit() for c in text) and
                            text not in ['主', '客', 'vs', '-', ':']):
                            if text not in teams:
                                teams.append(text)
                        if len(teams) >= 2:
                            break

                teams = teams[:2]

                if len(teams) >= 2:
                    matches.append({
                        'order': idx,
                        'match_id': match_id,
                        'date': date_str,
                        'league': league or '未知',
                        'time': match_time,
                        'status': status,
                        'home_team': teams[0],
                        'away_team': teams[1],
                        'score': "",
                        'score_home': "",
                        'score_away': "",
                        'actual_result': "",  # 未来比赛没有结果
                        'has_result': False
                    })

            except Exception as e:
                continue

        df = pd.DataFrame(matches)
        if not df.empty:
            df = df.sort_values('order').reset_index(drop=True)
        return df

    def fetch_future_matches_with_odds(self, date_str: str, progress_callback=None, log_callback=None) -> pd.DataFrame:
        """获取未来比赛并获取赔率数据"""
        if log_callback:
            self.set_log_callback(log_callback)

        self._log(f"🔮 获取 {date_str} 的未来比赛...")

        matches = self.fetch_future_matches(date_str)
        if matches.empty:
            self._log(f"⚠️ {date_str} 没有未来比赛")
            return pd.DataFrame()

        self._log(f"📝 找到 {len(matches)} 场未来比赛")

        match_list = []
        for idx, row in matches.iterrows():
            if progress_callback:
                progress_callback(idx+1, len(matches), date_str, f"{row['home_team']} vs {row['away_team']}")

            self._log(f"📊 获取赔率: {row['home_team']} vs {row['away_team']}")

            odds = self.fetch_all_odds(row['match_id'], log_callback)

            match_data = row.to_dict()
            match_data.update(odds)
            match_list.append(match_data)

            time.sleep(0.2)

        df = pd.DataFrame(match_list)
        if not df.empty:
            df = df.sort_values('order').reset_index(drop=True)

        self._log(f"✅ {date_str} 完成，共 {len(df)} 场")
        return df
class FeatureEngineer:
    """特征工程 - 增强版，处理维度问题和类别不平衡"""

    FEATURE_DIM = 85

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_features(self, match_data: Dict) -> np.ndarray:
        """提取特征 - 确保输出固定85维"""
        features = []

        # 1. 欧洲赔率特征 (35维)
        europe = match_data.get('europe', [])
        features.extend(self._europe_features(europe))

        # 2. 亚盘特征 (15维)
        asia = match_data.get('asia', [])
        features.extend(self._asia_features(asia))

        # 3. 让球特征 (10维)
        handicap = match_data.get('handicap', [])
        features.extend(self._handicap_features(handicap))

        # 4. 大小球特征 (12维)
        daxiao = match_data.get('daxiao', [])
        features.extend(self._daxiao_features(daxiao))

        # 5. 元特征 (13维)
        features.extend(self._meta_features(match_data))

        # 【关键修复】确保维度正好是85
        if len(features) > self.FEATURE_DIM:
            features = features[:self.FEATURE_DIM]
        elif len(features) < self.FEATURE_DIM:
            features.extend([0.0] * (self.FEATURE_DIM - len(features)))

        return np.array(features)

    def _europe_features(self, europe_odds):
        """欧洲赔率特征 - 固定35维"""
        if not europe_odds or len(europe_odds) == 0:
            return [0.0] * 35

        try:
            df = pd.DataFrame(europe_odds)
            features = []

            # 基础统计 (15维)
            for col in ['live_home', 'live_draw', 'live_away']:
                if col in df.columns and len(df) > 0:
                    features.extend([
                        float(df[col].mean()), 
                        float(df[col].std()) if len(df) > 1 else 0.0, 
                        float(df[col].min()), 
                        float(df[col].max()), 
                        float(df[col].median())
                    ])
                else:
                    features.extend([0.0] * 5)

            # 变化趋势 (6维)
            for col in ['change_home', 'change_draw', 'change_away']:
                if col in df.columns and len(df) > 0:
                    features.extend([
                        float(df[col].mean()),
                        (df[col] < 0).sum() / len(df) if len(df) > 0 else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0])

            # 凯利指数相关 (14维)
            if 'live_home' in df.columns and 'live_draw' in df.columns and 'live_away' in df.columns:
                total_prob = 1/df['live_home'] + 1/df['live_draw'] + 1/df['live_away']
                features.extend([
                    float(total_prob.mean()),
                    float(total_prob.std()) if len(total_prob) > 1 else 0.0,
                    float((1/df['live_home']).mean()),
                    float((1/df['live_draw']).mean()),
                    float((1/df['live_away']).mean()),
                    float((1/df['live_home']).std()) if len(df) > 1 else 0.0,
                    float((1/df['live_draw']).std()) if len(df) > 1 else 0.0,
                    float((1/df['live_away']).std()) if len(df) > 1 else 0.0,
                    (total_prob > 1.1).sum() / len(df),
                    (total_prob < 0.95).sum() / len(df),
                    float(df['live_home'].mean() / df['live_away'].mean()) if df['live_away'].mean() != 0 else 1.0,
                    float((df['live_home'] < df['live_away']).sum() / len(df)),
                    float(df['live_draw'].mean()),
                    float(len(df))
                ])
            else:
                features.extend([0.0] * 14)

            return features[:35]
        except Exception as e:
            return [0.0] * 35 * 35

    def _asia_features(self, asia_odds):
        """亚盘特征 - 固定15维"""
        if not asia_odds or len(asia_odds) == 0:
            return [0.0] * 15

        try:
            df = pd.DataFrame(asia_odds)
            features = []

            # 水位统计 (8维)
            for col in ['live_home', 'live_away']:
                if col in df.columns and len(df) > 0:
                    features.extend([
                        float(df[col].mean()), 
                        float(df[col].std()) if len(df) > 1 else 0.0, 
                        float(df[col].min()), 
                        float(df[col].max())
                    ])
                else:
                    features.extend([0.0] * 4)

            # 盘口分析 (7维)
            if 'live_handicap' in df.columns:
                handicaps = []
                for h in df['live_handicap']:
                    try:
                        h_str = str(h).replace('球', '').replace('半', '.5').replace('平', '0')
                        handicaps.append(float(h_str))
                    except:
                        handicaps.append(0)

                if len(handicaps) > 0:
                    features.extend([
                        float(np.mean(handicaps)),
                        float(np.std(handicaps)) if len(handicaps) > 1 else 0.0,
                        float(max(handicaps)),
                        float(min(handicaps)),
                        (np.array(handicaps) > 0).sum() / len(handicaps),
                        (np.array(handicaps) < 0).sum() / len(handicaps),
                        len(set(handicaps)) / len(handicaps) if len(handicaps) > 0 else 0
                    ])
                else:
                    features.extend([0.0] * 7)
            else:
                features.extend([0.0] * 7)

            return features[:15]
        except Exception as e:
            return [0.0] * 15 * 15

    def _handicap_features(self, handicap_odds):
        """让球特征 - 固定10维"""
        if not handicap_odds or len(handicap_odds) == 0:
            return [0.0] * 10

        try:
            df = pd.DataFrame(handicap_odds)
            features = []

            for col in ['live_home', 'live_draw', 'live_away']:
                if col in df.columns and len(df) > 0:
                    features.extend([
                        float(df[col].mean()), 
                        float(df[col].std()) if len(df) > 1 else 0.0, 
                        float(df[col].min()), 
                        float(df[col].max())
                    ])
                else:
                    features.extend([0.0] * 4)

            if 'live_home' in df.columns and 'live_away' in df.columns and len(df) > 0:
                features.extend([
                    (df['live_home'] < df['live_away']).sum() / len(df),
                    (df['live_home'] > df['live_away']).sum() / len(df)
                ])
            else:
                features.extend([0.0, 0.0])

            return features[:10]
        except Exception as e:
            return [0.0] * 10 * 10

    def _daxiao_features(self, daxiao_odds):
        """大小球特征 - 固定12维"""
        if not daxiao_odds or len(daxiao_odds) == 0:
            return [0.0] * 12

        try:
            df = pd.DataFrame(daxiao_odds)
            features = []

            for col in ['live_over', 'live_under']:
                if col in df.columns and len(df) > 0:
                    features.extend([
                        float(df[col].mean()), 
                        float(df[col].std()) if len(df) > 1 else 0.0, 
                        float(df[col].min()), 
                        float(df[col].max())
                    ])
                else:
                    features.extend([0.0] * 4)

            if 'live_line' in df.columns:
                lines = []
                for line in df['live_line']:
                    try:
                        line_str = str(line).replace('球', '').replace('半', '.5')
                        lines.append(float(line_str))
                    except:
                        lines.append(2.5)

                if len(lines) > 0:
                    features.extend([
                        float(np.mean(lines)),
                        float(np.std(lines)) if len(lines) > 1 else 0.0,
                        float(max(lines)),
                        float(min(lines))
                    ])
                else:
                    features.extend([0.0] * 4)
            else:
                features.extend([0.0] * 4)

            return features[:12]
        except Exception as e:
            return [0.0] * 12 * 12

    def _meta_features(self, match_data):
        """元特征 - 固定13维"""
        features = []

        league = match_data.get('league', '')
        tier = 3
        if any(x in league for x in ['英超','西甲','意甲','德甲','法甲','欧冠']): tier = 1
        elif any(x in league for x in ['荷甲','葡超','俄超','比甲','欧罗巴']): tier = 2
        features.append(float(tier))

        match_time = match_data.get('time', '15:00')
        try:
            hour = int(match_time.split(':')[0])
            features.extend([
                float(np.sin(2 * np.pi * hour / 24)),
                float(np.cos(2 * np.pi * hour / 24)),
                1.0 if 19 <= hour <= 23 else 0.0,
                1.0 if hour < 12 else 0.0
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])

        europe = match_data.get('europe', [])
        if europe and len(europe) > 0:
            try:
                df = pd.DataFrame(europe)
                if 'live_home' in df.columns and 'live_away' in df.columns:
                    avg_home = df['live_home'].mean()
                    avg_away = df['live_away'].mean()
                    features.extend([
                        float(1/avg_home) if avg_home > 0 else 0.5,
                        float(1/avg_away) if avg_away > 0 else 0.5,
                        float(avg_away / avg_home) if avg_home > 0 else 1.0,
                        float((avg_home < avg_away).sum() / len(df)) if len(df) > 0 else 0.5
                    ])
                else:
                    features.extend([0.5, 0.5, 1.0, 0.5])
            except:
                features.extend([0.5, 0.5, 1.0, 0.5])
        else:
            features.extend([0.5, 0.5, 1.0, 0.5])

        for odds_type in ['europe', 'asia', 'handicap', 'daxiao']:
            odds = match_data.get(odds_type, [])
            features.append(float(len(odds)) if odds else 0.0)

        return features[:13]

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
        """准备训练数据 - 增强版，处理类别不平衡"""
        X, y_labels, metadata = [], [], []

        # 调试统计
        total_rows = len(df)
        has_result_count = 0
        has_odds_count = 0
        valid_features_count = 0

        for _, row in df.iterrows():
            if not row.get('actual_result') or row['actual_result'] not in ['主胜', '平局', '客胜']:
                continue
            has_result_count += 1

            has_odds = any(len(row.get(ot, []) or []) > 0 for ot in ['europe', 'asia', 'handicap', 'daxiao'])
            if not has_odds:
                continue
            has_odds_count += 1

            features = self.extract_features(row.to_dict())
            valid_features_count += 1

            # 确保维度
            if len(features) != self.FEATURE_DIM:
                if len(features) > self.FEATURE_DIM:
                    features = features[:self.FEATURE_DIM]
                else:
                    features = np.concatenate([features, np.zeros(self.FEATURE_DIM - len(features))])

            X.append(features)

            result = row['actual_result']
            if result == '主胜':
                y_labels.append(0)
            elif result == '平局':
                y_labels.append(1)
            else:
                y_labels.append(2)

            metadata.append({
                'match_id': row['match_id'],
                'teams': f"{row['home_team']} vs {row['away_team']}",
                'date': row['date'],
                'league': row['league']
            })

        # 输出调试信息
        print(f"📊 数据过滤统计:")
        print(f"   总行数: {total_rows}")
        print(f"   有结果: {has_result_count}")
        print(f"   有赔率: {has_odds_count}")
        print(f"   有效特征: {valid_features_count}")

        if len(X) == 0:
            return np.array([]), np.array([]), []

        X = np.array(X)
        y_labels = np.array(y_labels)

        # 检查类别数量
        unique_classes = np.unique(y_labels)
        class_counts = np.bincount(y_labels, minlength=3)
        print(f"📊 类别分布: 主胜={class_counts[0]}, 平局={class_counts[1]}, 客胜={class_counts[2]}")

        if len(unique_classes) < 2:
            print(f"⚠️ 只有 {len(unique_classes)} 个类别，无法训练")
            return np.array([]), np.array([]), []

        # 数据增强
        min_count = min(class_counts[class_counts > 0])
        if min_count < 5:
            print(f"⚠️ 某些类别样本过少，正在数据增强...")
            X_new, y_new, meta_new = [], [], []

            for cls in range(3):
                mask = y_labels == cls
                X_cls = X[mask]
                y_cls = y_labels[mask]
                meta_cls = [metadata[i] for i in range(len(metadata)) if mask[i]]

                if len(X_cls) > 0 and len(X_cls) < 5:
                    repeat_times = (5 // len(X_cls)) + 1
                    X_cls = np.repeat(X_cls, repeat_times, axis=0)[:10]
                    y_cls = np.repeat(y_cls, repeat_times)[:10]
                    meta_cls = (meta_cls * repeat_times)[:10]

                if len(X_cls) > 0:
                    X_new.extend(X_cls)
                    y_new.extend(y_cls)
                    meta_new.extend(meta_cls)

            X = np.array(X_new)
            y_labels = np.array(y_new)
            metadata = meta_new
            print(f"📈 增强后: {len(X)} 个样本")

        # 转换为one-hot
        y_onehot = np.zeros((len(y_labels), 3))
        for i, label in enumerate(y_labels):
            y_onehot[i, label] = 1

        if not self.is_fitted:
            X = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X = self.scaler.transform(X)

        return X, y_onehot, metadata

    def transform(self, match_data: Dict) -> np.ndarray:
        features = self.extract_features(match_data)

        if len(features) != self.FEATURE_DIM:
            if len(features) > self.FEATURE_DIM:
                features = features[:self.FEATURE_DIM]
            else:
                features = np.concatenate([features, np.zeros(self.FEATURE_DIM - len(features))])

        if self.is_fitted:
            features = self.scaler.transform(features.reshape(1, -1))[0]
        return features

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_fitted': self.is_fitted}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']


class DeepLearningModel:
    """深度学习模型"""
    
    def __init__(self, input_dim=85):
        self.input_dim = input_dim
        self.dnn_model = None
        self.rf_model = None
        self.gbdt_model = None
        self.is_trained = False
        self.training_history = []
    
    def build_models(self):
        if TF_AVAILABLE:
            self.dnn_model = self._build_dnn()
        
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
        self.gbdt_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
    
    def _build_dnn(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100):
        if len(X) < 10:
            return False, "数据量不足"
        
        results = {}
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        if TF_AVAILABLE and self.dnn_model:
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            history = self.dnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                        epochs=epochs, batch_size=32, callbacks=callbacks, verbose=0)
            results['dnn'] = {
                'train_acc': max(history.history['accuracy']),
                'val_acc': max(history.history['val_accuracy'])
            }
            self.training_history.append(history.history)
        
        y_train_labels = np.argmax(y_train, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        
        self.rf_model.fit(X_train, y_train_labels)
        results['rf'] = {
            'train_acc': self.rf_model.score(X_train, y_train_labels),
            'val_acc': self.rf_model.score(X_val, y_val_labels)
        }
        
        self.gbdt_model.fit(X_train, y_train_labels)
        results['gbdt'] = {
            'train_acc': self.gbdt_model.score(X_train, y_train_labels),
            'val_acc': self.gbdt_model.score(X_val, y_val_labels)
        }
        
        self.is_trained = True
        return True, results
    
    def predict(self, X):
        if not self.is_trained:
            return None
        
        dnn_pred = None
        if TF_AVAILABLE and self.dnn_model:
            dnn_pred = self.dnn_model.predict(X.reshape(1, -1), verbose=0)[0]
        
        rf_pred = self.rf_model.predict_proba(X.reshape(1, -1))[0]
        gbdt_pred = self.gbdt_model.predict_proba(X.reshape(1, -1))[0]
        
        if dnn_pred is not None:
            ensemble_pred = 0.4 * dnn_pred + 0.3 * rf_pred + 0.3 * gbdt_pred
        else:
            ensemble_pred = 0.5 * rf_pred + 0.5 * gbdt_pred
        
        ensemble_pred = ensemble_pred / ensemble_pred.sum()
        
        labels = ['主胜', '平局', '客胜']
        pred_idx = np.argmax(ensemble_pred)
        
        return {
            'result': labels[pred_idx],
            'confidence': round(ensemble_pred[pred_idx] * 100, 2),
            'probabilities': {labels[i]: round(ensemble_pred[i] * 100, 2) for i in range(3)}
        }
    
    def _predict_scores(self, home_win_prob, draw_prob, away_win_prob):
        """预测最可能的3个比分"""
        scores = []

        if home_win_prob > 0.3:
            scores.extend([
                ('1-0', home_win_prob * 0.25),
                ('2-0', home_win_prob * 0.20),
                ('2-1', home_win_prob * 0.18),
                ('3-0', home_win_prob * 0.12),
                ('3-1', home_win_prob * 0.10),
            ])

        if draw_prob > 0.2:
            scores.extend([
                ('1-1', draw_prob * 0.40),
                ('0-0', draw_prob * 0.30),
                ('2-2', draw_prob * 0.20),
            ])

        if away_win_prob > 0.3:
            scores.extend([
                ('0-1', away_win_prob * 0.25),
                ('0-2', away_win_prob * 0.20),
                ('1-2', away_win_prob * 0.18),
                ('0-3', away_win_prob * 0.12),
                ('1-3', away_win_prob * 0.10),
            ])

        score_dict = {}
        for score, weight in scores:
            if score in score_dict:
                score_dict[score] += weight
            else:
                score_dict[score] = weight

        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_scores[:3]

        total_weight = sum(w for _, w in top3) if top3 else 1
        return [(score, round(w/total_weight*100, 2)) for score, w in top3]

    def _predict_total_goals(self, home_win_prob, draw_prob, away_win_prob):
        """预测最可能的3个总进球数"""
        goals = []

        goals.extend([
            (0, draw_prob * 0.15 + (home_win_prob + away_win_prob) * 0.05),
            (1, (home_win_prob + away_win_prob) * 0.25),
            (2, draw_prob * 0.30 + (home_win_prob + away_win_prob) * 0.20),
            (3, draw_prob * 0.35 + (home_win_prob + away_win_prob) * 0.30),
            (4, draw_prob * 0.15 + (home_win_prob + away_win_prob) * 0.25),
            (5, (home_win_prob + away_win_prob) * 0.10),
            (6, (home_win_prob + away_win_prob) * 0.05),
        ])

        sorted_goals = sorted(goals, key=lambda x: x[1], reverse=True)
        top3 = sorted_goals[:3]

        total_weight = sum(w for _, w in top3) if top3 else 1
        return [(str(goals), round(w/total_weight*100, 2)) for goals, w in top3]

    def save(self, name='model_v5'):
        path = os.path.join(CONFIG.MODEL_DIR, name)
        os.makedirs(path, exist_ok=True)
        
        if TF_AVAILABLE and self.dnn_model:
            self.dnn_model.save(os.path.join(path, 'dnn.h5'))
        
        with open(os.path.join(path, 'rf.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)
        with open(os.path.join(path, 'gbdt.pkl'), 'wb') as f:
            pickle.dump(self.gbdt_model, f)
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump({'is_trained': self.is_trained, 'training_history': self.training_history}, f)
        
        return True
    
    def load(self, name='model_v5'):
        path = os.path.join(CONFIG.MODEL_DIR, name)
        
        if not os.path.exists(path):
            return False
        
        try:
            if TF_AVAILABLE and os.path.exists(os.path.join(path, 'dnn.h5')):
                self.dnn_model = load_model(os.path.join(path, 'dnn.h5'))
            
            with open(os.path.join(path, 'rf.pkl'), 'rb') as f:
                self.rf_model = pickle.load(f)
            with open(os.path.join(path, 'gbdt.pkl'), 'rb') as f:
                self.gbdt_model = pickle.load(f)
            
            with open(os.path.join(path, 'config.pkl'), 'rb') as f:
                config = pickle.load(f)
                self.is_trained = config['is_trained']
                self.training_history = config['training_history']
            
            return True
        except:
            return False

# ==================== 主控制系统 ====================

class FootballPredictionSystem:
    """足球预测主系统"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.model = DeepLearningModel(input_dim=FeatureEngineer.FEATURE_DIM)
        self.data_file = CONFIG.DATA_FILE
        
        self.df = self._load_data()
        
        if self.model.load():
            scaler_path = os.path.join(CONFIG.MODEL_DIR, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.feature_engineer.load(scaler_path)
    
    def _load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                
                required_cols = ['match_id', 'date', 'league', 'time', 'home_team', 'away_team', 'actual_result']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = ''
                
                for col in ['europe', 'asia', 'daxiao', 'handicap']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x if isinstance(x, list) else [])
                    else:
                        df[col] = [[] for _ in range(len(df))]
                
                return df
            except Exception as e:
                print(f"加载数据失败: {e}")
        
        return pd.DataFrame(columns=[
            'match_id', 'date', 'league', 'time', 'status', 'home_team', 'away_team',
            'score', 'score_home', 'score_away', 'actual_result', 'has_result',
            'europe', 'asia', 'daxiao', 'handicap', 'order'
        ])
    
    def _save_data(self):
        try:
            if self.df.empty:
                with open(self.data_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                return True
            
            df_copy = self.df.copy()
            for col in ['europe', 'asia', 'daxiao', 'handicap']:
                if col in df_copy.columns:
                    def serialize_odds(x):
                        if isinstance(x, list):
                            return json.dumps(x, ensure_ascii=False)
                        elif isinstance(x, str):
                            try:
                                json.loads(x)
                                return x
                            except:
                                return json.dumps([], ensure_ascii=False)
                        else:
                            return json.dumps([], ensure_ascii=False)

                    df_copy[col] = df_copy[col].apply(serialize_odds)
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(df_copy.to_dict('records'), f, ensure_ascii=False, indent=2)

            # 调试信息
            print(f"💾 保存了 {len(df_copy)} 场比赛数据")
            for col in ['europe', 'asia', 'daxiao', 'handicap']:
                if col in df_copy.columns:
                    count = df_copy[col].apply(lambda x: len(json.loads(x)) > 0 if isinstance(x, str) else False).sum()
                    print(f"   {col}: {count} 场有数据")

            return True
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False
    
    def batch_collect_training_data(self, start_date: str, days: int = 7, progress_callback=None, log_callback=None):
        """批量收集训练数据"""
        new_df = self.collector.batch_fetch_history(start_date, days, progress_callback, log_callback)
        
        if new_df.empty:
            return False, "未获取到数据"
        
        if not self.df.empty and 'match_id' in self.df.columns:
            existing_ids = set(self.df['match_id'].tolist())
            new_df = new_df[~new_df['match_id'].isin(existing_ids)]
        
        if not new_df.empty:
            for col in self.df.columns:
                if col not in new_df.columns:
                    if col in ['europe', 'asia', 'daxiao', 'handicap']:
                        new_df[col] = [[] for _ in range(len(new_df))]
                    else:
                        new_df[col] = ['' for _ in range(len(new_df))]
            
            for col in new_df.columns:
                if col not in self.df.columns:
                    if col in ['europe', 'asia', 'daxiao', 'handicap']:
                        self.df[col] = [[] for _ in range(len(self.df))]
                    else:
                        self.df[col] = ['' for _ in range(len(self.df))]
            
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self._save_data()
            
            has_result = len(new_df[new_df['actual_result'] != ''])
            has_odds = len(new_df[new_df['europe'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)])
            
            return True, f"新增 {len(new_df)} 场（有结果: {has_result}场，有赔率: {has_odds}场），累计 {len(self.df)} 场"
        
        return True, "所有数据已是最新"

    def collect_future_matches(self, date_str: str, progress_callback=None, log_callback=None):
        """收集未来比赛用于预测"""
        new_df = self.collector.fetch_future_matches_with_odds(date_str, progress_callback, log_callback)

        if new_df.empty:
            return False, "未获取到未来比赛"

        # 检查是否已存在
        if not self.df.empty and 'match_id' in self.df.columns:
            existing_ids = set(self.df['match_id'].tolist())
            new_df = new_df[~new_df['match_id'].isin(existing_ids)]

        if not new_df.empty:
            # 确保列一致
            for col in self.df.columns:
                if col not in new_df.columns:
                    if col in ['europe', 'asia', 'daxiao', 'handicap']:
                        new_df[col] = [[] for _ in range(len(new_df))]
                    else:
                        new_df[col] = ['' for _ in range(len(new_df))]

            for col in new_df.columns:
                if col not in self.df.columns:
                    if col in ['europe', 'asia', 'daxiao', 'handicap']:
                        self.df[col] = [[] for _ in range(len(self.df))]
                    else:
                        self.df[col] = ['' for _ in range(len(self.df))]

            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self._save_data()

            has_odds = len(new_df[new_df['europe'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)])

            return True, f"新增 {len(new_df)} 场未来比赛（有赔率: {has_odds}场）"

        return True, "所有未来比赛已是最新"

    def train_model(self):
        """训练模型"""
        if self.df.empty or 'actual_result' not in self.df.columns:
            return False, "没有训练数据"
        
        train_df = self.df[self.df['actual_result'].isin(['主胜', '平局', '客胜'])].copy()
        
        if len(train_df) < CONFIG.MIN_TRAIN_SAMPLES:
            return False, f"需要至少{CONFIG.MIN_TRAIN_SAMPLES}场有结果的比赛，当前{len(train_df)}场"
        
        has_odds_count = 0
        for idx, row in train_df.iterrows():
            for odds_type in ['europe', 'asia', 'handicap', 'daxiao']:
                odds = row.get(odds_type, [])
                if odds and len(odds) > 0:
                    has_odds_count += 1
                    break
        
        if has_odds_count == 0:
            return False, f"有结果的比赛: {len(train_df)}场，但有赔率数据的: 0场。请确保成功获取了赔率信息。"
        
        X, y, metadata = self.feature_engineer.prepare_training_data(train_df)
        
        if len(X) == 0:
            return False, f"特征提取失败。有赔率的比赛: {has_odds_count}场，但无法提取有效特征。"
        
        self.model.build_models()
        success, results = self.model.train(X, y)
        
        if success:
            self.model.save()
            self.feature_engineer.save(os.path.join(CONFIG.MODEL_DIR, 'scaler.pkl'))
            return True, results
        
        return False, results
    
    def predict(self, match_data):
        """预测"""
        if not self.model.is_trained:
            return None, "模型未训练"
        
        features = self.feature_engineer.transform(match_data)
        result = self.model.predict(features)
        return result, None
    
    def get_stats(self):
        """获取统计"""
        if self.df.empty:
            return {'total': 0, 'trainable': 0, 'model_ready': self.model.is_trained}
        
        if 'actual_result' not in self.df.columns:
            trainable = 0
        else:
            trainable = len(self.df[self.df['actual_result'].isin(['主胜', '平局', '客胜'])])
        
        return {
            'total': len(self.df),
            'trainable': trainable,
            'model_ready': self.model.is_trained
        }

# ==================== Streamlit UI ====================

def main():
    st.set_page_config(page_title="⚽ 足球智能预测系统 v5.2", layout="wide")
    
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">⚽ 足球智能预测系统 v5.2<br><small>优化赔率提取版</small></div>', unsafe_allow_html=True)
    
    if 'system' not in st.session_state:
        with st.spinner("系统初始化中..."):
            st.session_state['system'] = FootballPredictionSystem()
    
    system = st.session_state['system']
    stats = system.get_stats()
    
    with st.sidebar:
        st.header("🎛️ 控制面板")
        
        st.subheader("📚 批量获取训练数据")
        col_date, col_days = st.columns([2, 1])
        with col_date:
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=7))
        with col_days:
            days = st.number_input("天数", min_value=1, max_value=30, value=7)
        
        if 'fetch_logs' not in st.session_state:
            st.session_state['fetch_logs'] = []
        
        log_area = st.empty()
        if st.session_state['fetch_logs']:
            with log_area.container():
                st.markdown("📋 **获取日志**")
                log_html = "<div style='height: 120px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 5px; padding: 8px; background-color: #f8f9fa; font-family: monospace; font-size: 10px; line-height: 1.6;'>"
                for log in st.session_state['fetch_logs'][-50:]:
                    color = '#0066cc' if '📡' in log else '#28a745' if '✅' in log else '#333'
                    log_html += f"<div style='color: {color}; margin-bottom: 2px;'>{log}</div>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)
        
        # 获取未来比赛
        st.subheader("获取未来比赛")
        future_date = st.date_input("选择比赛日期", datetime.now() + timedelta(days=1))

        if st.button("获取未来比赛", use_container_width=True, type="primary"):
            st.session_state['fetch_logs'] = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, date_str, status):
                progress_bar.progress(current / total if total > 0 else 0)
                status_text.text(f"[{current}/{total}] {status}")

            def add_log(message):
                st.session_state['fetch_logs'].append(message)
                if len(st.session_state['fetch_logs']) > 100:
                    st.session_state['fetch_logs'] = st.session_state['fetch_logs'][-100:]

            with st.spinner("获取未来比赛中..."):
                success, msg = system.collect_future_matches(
                    future_date.strftime('%Y-%m-%d'),
                    update_progress,
                    add_log
                )

            progress_bar.empty()
            status_text.empty()

            if success:
                st.success(msg)
                st.balloons()
            else:
                st.error(msg)

            st.rerun()

        st.markdown("---")

        if st.button("🚀 批量获取训练数据", use_container_width=True, type="primary"):
            st.session_state['fetch_logs'] = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current_day, total_days, date_str, status):
                progress_bar.progress(current_day / total_days)
                status_text.text(f"[{current_day}/{total_days}] {date_str}: {status}")
            
            def add_log(message):
                st.session_state['fetch_logs'].append(message)
                if len(st.session_state['fetch_logs']) > 100:
                    st.session_state['fetch_logs'] = st.session_state['fetch_logs'][-100:]
            
            with st.spinner("获取中..."):
                success, msg = system.batch_collect_training_data(
                    start_date.strftime('%Y-%m-%d'),
                    int(days),
                    update_progress,
                    add_log
                )
            
            progress_bar.empty()
            status_text.empty()
            
            if success:
                st.success(msg)
                st.balloons()
            else:
                st.error(msg)
            
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("🧠 模型训练")
        trainable_count = stats['trainable']
        
        if trainable_count >= CONFIG.MIN_TRAIN_SAMPLES:
            st.success(f"✅ 可训练数据: {trainable_count}场")
            if st.button("开始训练模型", use_container_width=True, type="primary"):
                with st.spinner("训练中..."):
                    success, result = system.train_model()
                    if success:
                        st.success("训练完成！")
                        st.json(result)
                    else:
                        st.error(result)
        else:
            st.warning(f"⚠️ 需要{CONFIG.MIN_TRAIN_SAMPLES}场，当前{trainable_count}场")
            st.progress(trainable_count / CONFIG.MIN_TRAIN_SAMPLES if CONFIG.MIN_TRAIN_SAMPLES > 0 else 0)
        
        st.markdown("---")
        
        if st.button("💾 保存数据", use_container_width=True):
            if system._save_data():
                st.success("已保存")
        
        if st.button("🗑️ 清空数据", use_container_width=True):
            system.df = pd.DataFrame(columns=[
                'match_id', 'date', 'league', 'time', 'status', 'home_team', 'away_team',
                'score', 'score_home', 'score_away', 'actual_result', 'has_result',
                'europe', 'asia', 'daxiao', 'handicap', 'order'
            ])
            system._save_data()
            st.success("已清空")
    
    tabs = st.tabs(["📊 数据概览", "🎯 预测中心"])
    
    with tabs[0]:
        st.subheader("训练数据概览")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("总比赛数", stats['total'])
        with cols[1]:
            st.metric("可训练数据", stats['trainable'])
        with cols[2]:
            st.metric("训练门槛", CONFIG.MIN_TRAIN_SAMPLES)
        with cols[3]:
            st.metric("模型状态", "✅ 就绪" if stats['model_ready'] else "❌ 未训练")
        
        if not system.df.empty:
            st.markdown("---")
            
            display_df = system.df.copy()
            if 'actual_result' in display_df.columns:
                display_df = display_df[display_df['actual_result'].isin(['主胜', '平局', '客胜'])]
            
            display_cols = ['date', 'league', 'home_team', 'away_team', 'score', 'actual_result']
            display_cols = [c for c in display_cols if c in display_df.columns]

            # 添加赔率状态列
            def get_odds_status(row):
                odds_types = []
                for ot in ['europe', 'asia', 'handicap', 'daxiao']:
                    if ot in row and isinstance(row[ot], list) and len(row[ot]) > 0:
                        odds_types.append(ot[:2])
                return ','.join(odds_types) if odds_types else '无'

            if not display_df.empty:
                display_df['odds'] = display_df.apply(get_odds_status, axis=1)
                display_cols.append('odds')
            
            if not display_df.empty:
                if 'order' in display_df.columns:
                    display_df = display_df.sort_values(['date', 'order'])
                
                st.dataframe(display_df[display_cols], use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("赔率数据统计")
                odds_stats = {}
                for odds_type in ['europe', 'asia', 'handicap', 'daxiao']:
                    if odds_type in display_df.columns:
                        count = display_df[odds_type].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
                        odds_stats[odds_type] = count
                
                if odds_stats:
                    st.write(odds_stats)
            else:
                st.info("没有符合条件的数据")
        else:
            st.info("暂无数据，请使用侧边栏获取训练数据")
    
    with tabs[1]:
        st.subheader("比赛预测")
        
        if not system.df.empty and stats['model_ready']:
            future_matches = system.df[system.df['actual_result'] == ''] if 'actual_result' in system.df.columns else system.df
            
            if not future_matches.empty:
                selected = st.selectbox(
                    "选择比赛进行预测",
                    future_matches.to_dict('records'),
                    format_func=lambda x: f"{x.get('date', 'N/A')} | {x.get('league', 'N/A')} | {x.get('home_team', 'N/A')} vs {x.get('away_team', 'N/A')}"
                )
                
                if selected:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {selected.get('home_team', 'N/A')} 🆚 {selected.get('away_team', 'N/A')}")
                        st.write(f"**联赛:** {selected.get('league', 'N/A')}")
                        
                        europe_odds = selected.get('europe', [])
                        has_odds = europe_odds and len(europe_odds) > 0
                        if has_odds:
                            st.write(f"**赔率数据:** ✅ 已获取 ({len(europe_odds)}家公司)")
                        else:
                            st.warning("⚠️ 暂无赔率数据")
                    
                    with col2:
                        if has_odds:
                            if st.button("🔮 开始预测", type="primary", use_container_width=True):
                                result, error = system.predict(selected)
                                if error:
                                    st.error(error)
                                else:
                                    # 胜平负预测
                                    st.success(f"**预测结果: {result['result']}**")
                                    st.write(f"**置信度:** {result['confidence']}%")

                                    # 显示前3个最可能的结果
                                    st.markdown("**胜平负概率 Top 3:**")
                                    for i, (res, conf) in enumerate(result['top3_results'], 1):
                                        st.write(f"{i}. {res}: {conf}%")

                                    # 比分预测
                                    st.markdown("**最可能比分 Top 3:**")
                                    for i, (score, prob) in enumerate(result['score_predictions'], 1):
                                        st.write(f"{i}. {score}: {prob}%")

                                    # 总进球数预测
                                    st.markdown("**总进球数 Top 3:**")
                                    for i, (goals, prob) in enumerate(result['total_goals_predictions'], 1):
                                        st.write(f"{i}. {goals}球: {prob}%")
                        else:
                            st.info("请先获取赔率数据")
            else:
                st.info("当前没有未预测的比赛")
        else:
            if not stats['model_ready']:
                st.warning("⚠️ 模型未训练")
            else:
                st.info("暂无比赛数据")

if __name__ == "__main__":
    main()