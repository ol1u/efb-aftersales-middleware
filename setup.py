import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    raise Exception("Python 3.7 or higher is required. Your version is %s." % sys.version)

__version__ = ""
exec(open("efb_aftersales_middleware/__version__.py", encoding="utf-8").read())

setup(
    name="efb-aftersales-middleware",
    packages=find_packages(exclude=["tests", "tests.*"]),
    version=__version__,
    description="EFB middleware that turns WeChat after-sales group messages into tracked tickets.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="AGPLv3+",
    python_requires=">=3.7",
    install_requires=[
        "ehforwarderbot>=2.1.0",
        "requests",
        "ruamel.yaml",
    ],
    extras_require={"tests": ["pytest"]},
    entry_points={
        "ehforwarderbot.middleware": "ol1u.aftersales = efb_aftersales_middleware:AfterSalesMiddleware",
    },
)
