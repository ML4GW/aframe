"""
Tools for deploying helm charts
"""
import logging
import subprocess
import time
from typing import Dict, List, Optional

import kr8s

CHART_REPO = "https://github.com/EthanMarx/lightray/releases/download/"


def authenticate():
    subprocess.run(
        ["kubectl", "cluster-info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# used to monkey patch kr8s api refresh method;
# see https://github.com/kr8s-org/kr8s/issues/214
async def auth(self):
    """
    Replacement reauthenticate method that
    uses kubectl to refresh the OIDC token
    """
    authenticate()
    await self._load_kubeconfig()


class HelmChart:
    def __init__(self, chart_url: str, release: str):
        self.chart_url = chart_url
        self.release = release
        base_cmd = ["helm", "install", self.release, self.chart_url]

        self.base_cmd = base_cmd

        api = kr8s.api()
        api.auth.reauthenticate = auth.__get__(api.auth, type(api.auth))
        self.api = api

    def install(self):
        logging.info(f"Installing chart from {self.chart_url}")
        try:
            subprocess.run(self.base_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing helm chart: {e}")
            raise

    def build_command(self, values: Dict[str, str]) -> List[str]:
        for k, v in values.items():
            self.base_cmd += ["--set", f"{k}={v}"]

    def uninstall(self):
        subprocess.run(["helm", "uninstall", self.release], check=True)

    def wait(self):
        raise NotImplementedError

    def __enter__(self):
        self.install()
        self.wait()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()
        super().__exit__(exc_type, exc_value, traceback)


class RayCluster(HelmChart):
    def __init__(
        self,
        release: str,
        chart_path: Optional[str] = None,
        chart_version: str = "0.1.1",
    ):
        # if no chart path is provided, use the chart
        # in the github repo
        if chart_path is None:
            chart_path = f"{CHART_REPO}/ray-cluster-{chart_version}"
            chart_path += "/ray-cluster-{chart_version}.tgz"
        super().__init__(chart_path, release)

    def get_ip(self):
        services = kr8s.get("service")
        services = [s for s in services if s.name.startswith(self.release)]
        service = [s for s in services if s.spec["type"] == "LoadBalancer"][0]
        service.refresh()
        return service.status.loadBalancer.ingress[0].ip

    def get_pods(self):
        pods = kr8s.get("pod")
        # filter for pods related to this release
        # and that aren't terminating from a previous run
        pods = [
            p
            for p in pods
            if p.name.startswith(self.release)
            and p.status.phase in ["Pending", "Running"]
        ]

        head = [p for p in pods if "head" in p.name][0]
        workers = [p for p in pods if "worker" in p.name]
        return head, workers

    def wait(self):
        # get pods related to this release
        head, workers = self.get_pods()

        # wait for pods to be ready;
        # can subclass to define "readiness"
        ready = False
        while not ready:
            ready = any([p.ready() for p in workers])
            ready = ready and head.ready()
            time.sleep(2)
