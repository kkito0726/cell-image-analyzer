from setuptools import find_packages, setup


# requirements.txtを読み込む関数
def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()


setup(
    name="pyCell",  # パッケージ名（pip listで表示される）
    version="1.0.0",  # バージョン
    description="顕微鏡画像上の細胞についての解析をするライブラリ",  # 説明
    author="kentaro kito",  # 作者名
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=parse_requirements("requirements.txt"),
    license="MIT",  # ライセンス
)
