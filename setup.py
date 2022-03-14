from distutils.core import setup

setup(
    name="brats",
    version="0.2",
    description="""Final Project TM10007ML""",
    license="Apache 2.0 License",
    author="Martijn Starmans, Karin van Garderen, Hakim Achterberg",
    author_email="m.starmans@erasmusmc.nl",
    include_package_data=True,
    packages=[
        "brats",
        "hn",
        "adni"
    ],
)
