from world_models.helpers.jepa_helper import JEPAHelper


class TestJEPAHelper:
    def test_initialization(self):
        embed_dim = 256
        num_negatives = 10
        helper = JEPAHelper(embed_dim, num_negatives)
        assert helper.embed_dim == embed_dim
        assert helper.num_negatives == num_negatives
