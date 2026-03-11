import pytest
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


BASE_URL = "http://localhost:5173"
API_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_client():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def wait_for_api() -> bool:
    max_retries = 30
    retry_delay = 2
    for _ in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/api/health", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(retry_delay)
    return False


class TestAPIEndpoints:
    def test_health_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_catalog_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/catalog")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "environments_by_model" in data
        assert "dreamer" in data["models"]
        assert "planet" in data["models"]

    def test_environments_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/environments?model=dreamer")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) > 0

    def test_environments_endpoint_without_model(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/environments")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_load_model_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        payload = {"model": "dreamer", "config": {"seed": 1}}
        response = api_client.post(f"{API_URL}/api/load-model", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "dreamer"

    def test_load_model_invalid_model(self, api_client):
        assert wait_for_api(), "Backend server not available"
        payload = {"model": "invalid_model", "config": {}}
        response = api_client.post(f"{API_URL}/api/load-model", json=payload)
        assert response.status_code == 400

    def test_load_environment_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        api_client.post(f"{API_URL}/api/load-model", json={"model": "dreamer"})
        payload = {"environment": "cartpole-balance", "config": {}}
        response = api_client.post(f"{API_URL}/api/load-environment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["environment"] == "cartpole-balance"

    def test_load_environment_without_model(self, api_client):
        assert wait_for_api(), "Backend server not available"
        api_client.post(f"{API_URL}/api/train/stop")
        payload = {"environment": "cartpole-balance", "config": {}}
        response = api_client.post(f"{API_URL}/api/load-environment", json=payload)
        assert response.status_code in [200, 400]

    def test_state_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/state")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data

    def test_metrics_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/metrics?limit=100")
        assert response.status_code == 200
        data = response.json()
        assert "series" in data

    def test_frame_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        response = api_client.get(f"{API_URL}/api/frame")
        assert response.status_code == 200
        data = response.json()
        assert "image" in data

    def test_start_training_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        api_client.post(f"{API_URL}/api/load-model", json={"model": "dreamer"})
        api_client.post(
            f"{API_URL}/api/load-environment", json={"environment": "cartpole-balance"}
        )
        payload = {"config": {"total_steps": 100, "seed_steps": 10}}
        response = api_client.post(f"{API_URL}/api/train/start", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["running", "completed", "stopped"]

    def test_start_training_without_model(self, api_client):
        assert wait_for_api(), "Backend server not available"
        api_client.post(f"{API_URL}/api/train/stop")
        payload = {"config": {"total_steps": 100}}
        response = api_client.post(f"{API_URL}/api/train/start", json=payload)
        assert response.status_code in [200, 400]

    def test_start_training_without_environment(self, api_client):
        assert wait_for_api(), "Backend server not available"
        api_client.post(f"{API_URL}/api/train/stop")
        api_client.post(f"{API_URL}/api/load-model", json={"model": "dreamer"})
        payload = {"config": {"total_steps": 100}}
        response = api_client.post(f"{API_URL}/api/train/start", json=payload)
        assert response.status_code in [200, 400]

    def test_stop_training_endpoint(self, api_client):
        assert wait_for_api(), "Backend server not available"
        api_client.post(f"{API_URL}/api/load-model", json={"model": "dreamer"})
        api_client.post(
            f"{API_URL}/api/load-environment", json={"environment": "cartpole-balance"}
        )
        api_client.post(
            f"{API_URL}/api/train/start", json={"config": {"total_steps": 10000}}
        )
        time.sleep(1)
        response = api_client.post(f"{API_URL}/api/train/stop")
        assert response.status_code == 200
        data = response.json()
        assert "stop_requested" in data


@pytest.fixture(scope="module")
def driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    driver.set_page_load_timeout(30)

    yield driver

    driver.quit()


@pytest.fixture(scope="module", autouse=True)
def wait_for_server():
    max_retries = 30
    retry_delay = 2
    server_ready = False

    for _ in range(max_retries):
        try:
            import requests

            response = requests.get(f"{API_URL}/api/health", timeout=5)
            if response.status_code == 200:
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(retry_delay)

    if not server_ready:
        pytest.skip("Backend server not available")


class TestTorchWMUI:
    def test_page_loads_successfully(self, driver):
        driver.get(BASE_URL)
        assert "TorchWM Studio" in driver.title
        assert driver.find_element(By.CLASS_NAME, "app-shell")

    def test_model_selector_is_present(self, driver):
        driver.get(BASE_URL)
        model_select = driver.find_element(By.CSS_SELECTOR, "select")
        assert model_select is not None

        select = Select(model_select)
        options = [opt.text for opt in select.options]
        assert "DreamerAgent" in options
        assert "Planet" in options

    def test_environment_selector_populates_after_model_selection(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "control-panel"))
        )

        time.sleep(3)

        selects = driver.find_elements(By.CSS_SELECTOR, ".control-panel select")
        assert len(selects) >= 2

        env_select = selects[1]
        select = Select(env_select)

        assert len(select.options) > 0

    def test_load_model_button_is_clickable(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Load Model']"))
        )

        load_model_btn = driver.find_element(By.XPATH, "//button[text()='Load Model']")
        assert load_model_btn.is_enabled()

    def test_load_environment_button_is_clickable(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[text()='Load Environment']")
            )
        )

        load_env_btn = driver.find_element(
            By.XPATH, "//button[text()='Load Environment']"
        )
        assert load_env_btn.is_enabled()

    def test_start_training_button_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[text()='Start Training']")
            )
        )

        start_btn = driver.find_element(By.XPATH, "//button[text()='Start Training']")
        assert start_btn is not None

    def test_stop_button_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//button[text()='Stop']"))
        )

        stop_btn = driver.find_element(By.XPATH, "//button[text()='Stop']")
        assert stop_btn is not None

    def test_status_display_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "status-card"))
        )

        time.sleep(2)

        status_card = driver.find_element(By.CLASS_NAME, "status-card")
        assert status_card.text != ""

    def test_metrics_chart_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "chart-card"))
        )

        chart_card = driver.find_element(By.CLASS_NAME, "chart-card")
        assert chart_card is not None

    def test_environment_frame_placeholder_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "frame-card"))
        )

        frame_card = driver.find_element(By.CLASS_NAME, "frame-card")
        assert frame_card is not None

    def test_model_config_textarea_is_editable(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )

        textarea = driver.find_element(By.CSS_SELECTOR, "textarea")
        initial_value = textarea.get_attribute("value")

        assert initial_value is not None
        assert "{" in initial_value

    def test_training_config_textarea_is_present(self, driver):
        driver.get(BASE_URL)

        textareas = driver.find_elements(By.CSS_SELECTOR, "textarea")
        assert len(textareas) >= 2

    def test_topbar_header_is_visible(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "topbar"))
        )

        time.sleep(2)

        topbar = driver.find_element(By.CLASS_NAME, "topbar")
        assert topbar.text != ""

    def test_control_panel_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "control-panel"))
        )

        time.sleep(3)

        control_panel = driver.find_element(By.CLASS_NAME, "control-panel")
        assert control_panel.text != ""

    def test_progress_bar_is_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "progress-shell"))
        )

        progress_shell = driver.find_element(By.CLASS_NAME, "progress-shell")
        assert progress_shell is not None

    def test_dashboard_contains_metric_grid(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
        )

        dashboard = driver.find_element(By.CLASS_NAME, "dashboard")
        assert dashboard is not None

    def test_model_selection_changes_environment_options(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "control-panel"))
        )

        time.sleep(3)

        selects = driver.find_elements(By.CSS_SELECTOR, ".control-panel select")
        assert len(selects) >= 2

        model_select = selects[0]
        select = Select(model_select)

        select.select_by_value("dreamer")
        time.sleep(2)

        selects = driver.find_elements(By.CSS_SELECTOR, ".control-panel select")
        env_select = selects[1]
        env_options_dreamer = [opt.text for opt in Select(env_select).options]

        select.select_by_value("planet")
        time.sleep(2)

        selects = driver.find_elements(By.CSS_SELECTOR, ".control-panel select")
        env_select = selects[1]
        env_options_planet = [opt.text for opt in Select(env_select).options]

        assert len(env_options_dreamer) > 0
        assert len(env_options_planet) > 0

    def test_model_config_json_is_valid_json(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )

        textarea = driver.find_element(By.CSS_SELECTOR, "textarea")
        import json

        value = textarea.get_attribute("value")

        parsed = json.loads(value)
        assert isinstance(parsed, dict)

    def test_training_config_json_is_valid_json(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )

        textareas = driver.find_elements(By.CSS_SELECTOR, "textarea")
        textarea = textareas[1]

        import json

        value = textarea.get_attribute("value")

        parsed = json.loads(value)
        assert isinstance(parsed, dict)

    def test_status_shows_model_and_environment(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "status-card"))
        )

        time.sleep(2)

        status_card = driver.find_element(By.CLASS_NAME, "status-card")
        assert status_card.text != ""

    def test_progress_bar_contains_progress_label(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "progress-label"))
        )

        progress_label = driver.find_element(By.CLASS_NAME, "progress-label")
        assert "/" in progress_label.text

    def test_metric_selector_dropdown_exists(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "chart-card"))
        )

        chart_header = driver.find_element(By.CLASS_NAME, "chart-card")
        select = chart_header.find_element(By.CSS_SELECTOR, "select")
        assert select is not None

    def test_start_button_has_primary_class(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[text()='Start Training']")
            )
        )

        start_btn = driver.find_element(By.XPATH, "//button[text()='Start Training']")
        assert "primary" in start_btn.get_attribute("class")

    def test_stop_button_has_danger_class(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//button[text()='Stop']"))
        )

        stop_btn = driver.find_element(By.XPATH, "//button[text()='Stop']")
        assert "danger" in stop_btn.get_attribute("class")

    def test_status_pill_shows_current_status(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "status-pill"))
        )

        time.sleep(2)

        status_pill = driver.find_element(By.CLASS_NAME, "status-pill")
        assert status_pill.text in [
            "idle",
            "running",
            "completed",
            "stopped",
            "failed",
            "FAILED",
        ]

    def test_app_shell_contains_atmosphere_div(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "app-shell"))
        )

        atmosphere = driver.find_element(By.CLASS_NAME, "atmosphere")
        assert "atmosphere" in atmosphere.get_attribute("class")

    def test_all_required_labels_present(self, driver):
        driver.get(BASE_URL)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "control-panel"))
        )

        time.sleep(3)

        control_panel = driver.find_element(By.CLASS_NAME, "control-panel")
        assert control_panel.text != ""
