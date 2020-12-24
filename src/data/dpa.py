DATA_ROOT = "/data/DPA"


class DPASet:
    BASE_URL = ""
    TRACE_URLS = {}
    INDEX_URL = {}

    @classmethod
    def get_sha_sums(cls):
        return {**cls.INDEX_URL, **cls.TRACE_URLS}

    @classmethod
    def get_url(cls, file_name):
        return f"{cls.BASE_URL}/{file_name}"


class DPA4(DPASet):
    BASE_URL = "http://www.dpacontest.org/v4/traces/rsm"

    TRACE_URLS = {
        "DPA_contestv4_rsm_00000.zip": "e858af4dd6d662e41ce0c6dea02ec9eb6036c954",
        "DPA_contestv4_rsm_10000.zip": "620e02848d592cc3f93d96d53d1b4490afe74684",
        "DPA_contestv4_rsm_20000.zip": "b59c7cc1f395879c6a2cc68098615fd8fc5f373b",
        "DPA_contestv4_rsm_30000.zip": "cdbde49c4fd4482fecac731120a249bd59389309",
        "DPA_contestv4_rsm_40000.zip": "0574464273a1e382c1d282bca72f6740a924a6d5",
        "DPA_contestv4_rsm_50000.zip": "d6d95c46e90bce86bb043242062b28305eb23eb8",
        "DPA_contestv4_rsm_60000.zip": "979a8039b6f382bdefeaa6dbc2c493d202f92163",
        "DPA_contestv4_rsm_70000.zip": "66797d0f49d808c739183e2580df750b78238b41",
        "DPA_contestv4_rsm_80000.zip": "9cd564506907555db037b3a3661191d4b55225ea",
        "DPA_contestv4_rsm_90000.zip": "f37e7e08561b69c41000e36773e696b52dfd7b90"
    }

    INDEX_URL = {
        "dpav4_rsm_index": "792e966a60898e0b8fcb96f6b52ae42228f14970"
    }

    ROOT_RAW = f"{DATA_ROOT}/DPAv40/RAW"


class DPA4_2(DPASet):
    BASE_URL = "http://www.dpacontest.org/v4/traces/v4_2"

    TRACE_URLS = {
        "DPA_contestv4_2_k00.zip": "f711206b413b8d02f595d5861996ff61a1711f3d",
        "DPA_contestv4_2_k01.zip": "558020ee66c9948be8e08ea41e9fb7389d3f9db3",
        "DPA_contestv4_2_k02.zip": "1f1c539b9a2461994a126240b6cc8cf51193b4e5",
        "DPA_contestv4_2_k03.zip": "958827acf7dc9740715fe2d76f80d3f5e2e5237e",
        "DPA_contestv4_2_k04.zip": "74f3e7dbdaa9fe18660f6fbf462423a758e998bf",
        "DPA_contestv4_2_k05.zip": "5535342a4131d2881d23b4781a033efb89a4279f",
        "DPA_contestv4_2_k06.zip": "8bc2d9b322319bef4dc32fdce19357b3c2faf109",
        "DPA_contestv4_2_k07.zip": "aed1516046a6b6e4a80b3bec2190f7fbf60945e5",
        "DPA_contestv4_2_k08.zip": "25ec6a41fbbbca9829d1af51a133e01f46ba2abe",
        "DPA_contestv4_2_k09.zip": "1c96aa67e2f26edcd38ad152953c6059a1343bdc",
        "DPA_contestv4_2_k10.zip": "82365e174dd6ffb2c61e52154ed73f10b04f800d",
        "DPA_contestv4_2_k11.zip": "354d13867bf83d44e53a257e215d58572a4c9088",
        "DPA_contestv4_2_k12.zip": "b67e99c4aae59a6603bfd1f8183b15f1f4794d05",
        "DPA_contestv4_2_k13.zip": "e04f746fbc3bea19a2940250b10a3715a6964242",
        "DPA_contestv4_2_k14.zip": "77a9b52b0b8604a351a6af87812eb6306cfb1dfa",
        "DPA_contestv4_2_k15.zip": "1fe1086a8e05044363fc426a58e64bf6df61a658"
    }

    INDEX_URL = {
        "dpav4_2_index": "792e966a60898e0b8fcb96f6b52ae42228f14970"
    }

    ROOT_RAW = f"{DATA_ROOT}/DPAv42/RAW"
