import numpy as np
import pytest

from rles_world_model.routeb.market import MINUTE_NS
from rles_world_model.routeb.prior_forecast import (
    build_prior_alignment_table,
    write_prior_alignment_table,
)


def _fixture():
    anchors=np.asarray([[10,11,12],[20,21,22]],dtype=np.int64)*MINUTE_NS
    targets=anchors+MINUTE_NS
    shape=(*anchors.shape,5)
    logits=np.zeros((*anchors.shape,2,4),dtype=np.float32)
    return dict(
        anchor_timestamps_ns=anchors,
        target_timestamps_ns=targets,
        target_features=np.ones(shape,dtype=np.float32),
        valid_mask=np.asarray([[True,False,True],[True,True,False]]),
        prior_mean=np.zeros(shape,dtype=np.float32),
        prior_scale=np.ones(shape,dtype=np.float32),
        posterior_mean=np.full(shape,0.5,dtype=np.float32),
        posterior_scale=np.ones(shape,dtype=np.float32)*2,
        prior_logits=logits,
        posterior_logits=logits+1,
    )


def test_prior_alignment_table_retains_anchor_target_mask_and_posterior(tmp_path):
    table=build_prior_alignment_table(**_fixture())
    assert table["anchor_timestamps_ns"].shape==(6,)
    assert table["valid_mask"].tolist()==[True,False,True,True,True,False]
    assert table["prior_logits"].shape==(6,2,4)
    assert np.all(table["target_timestamps_ns"]
                  ==table["anchor_timestamps_ns"]+MINUTE_NS)
    path,digest=write_prior_alignment_table(tmp_path/"alignment.npz",table)
    assert len(digest)==64
    with np.load(path,allow_pickle=False) as artifact:
        assert set(artifact.files)==set(table)
        np.testing.assert_array_equal(artifact["posterior_mean"],table["posterior_mean"])


def test_valid_prior_alignment_rejects_misaligned_future_or_bad_scales():
    values=_fixture()
    values["target_timestamps_ns"]=values["target_timestamps_ns"].copy()
    values["target_timestamps_ns"][0,0]+=MINUTE_NS
    with pytest.raises(ValueError,match="next one-minute"):
        build_prior_alignment_table(**values)
    values=_fixture()
    values["prior_scale"]=values["prior_scale"].copy()
    values["prior_scale"][1,1,0]=0
    with pytest.raises(ValueError,match="positive scales"):
        build_prior_alignment_table(**values)
