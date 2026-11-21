from copy import deepcopy
from pathlib import Path

import pytest

from infrastructure.truba import refactor_reference as reference


def fixture():
    environment={k:"fixed" for k in (
        "platform","architecture","cpu_model","python","packages","torch_build",
        "numpy_build","scipy_build","torch_parallel_info","thread_environment","runtime_threadpools")}
    document={"environment":environment,"physical_backend_attempts":0,
        "first_adaptive":{"capture":{"reconstructed":{"candidate":[1.0,2.0]},"replay_outcome":{}}},
        "mathematical_checkpoints":{"terminal":{"score":1.0},"accounting":{"observations":40}}}
    environment['runtime_threadpools']=[{'filepath':'numpy.so','version':'one','num_threads':20},
                                      {'filepath':'scipy.so','version':'two','num_threads':20}]
    document['environment_signature_schema']='runtime-library-set-v2'
    document['environment_signature']=reference.environment_signature(environment)
    document['scientific_hash']=reference.digest(reference.numerical_payload(document))
    return document


@pytest.mark.parametrize('change',['none','candidate','terminal','accounting','environment','tamper'])
def test_exact_same_environment_gate_detects_real_changes(tmp_path:Path,change:str):
    first=fixture();second=deepcopy(first)
    if change=='candidate':second['first_adaptive']['capture']['reconstructed']['candidate'][0]+=0.00001
    if change=='terminal':second['mathematical_checkpoints']['terminal']['score']+=0.00001
    if change=='accounting':second['mathematical_checkpoints']['accounting']['observations']-=1
    if change=='environment':second['environment']['python']='changed'
    if change=='tamper':second['mathematical_checkpoints']['terminal']['score']+=1
    if change!='tamper':
        second['scientific_hash']=reference.digest(reference.numerical_payload(second))
        second['environment_signature']=reference.environment_signature(second['environment'])
    a=tmp_path/'a.json.gz';b=tmp_path/'b.json.gz'
    reference.write_gzip(a,first);reference.write_gzip(b,second)
    result=reference.compare_files(a,b)
    assert result['scientific_exact_equal']==(change in ('none','environment'))
    assert result['same_environment']==(change!='environment')
    assert result['artifacts_self_consistent']==(change!='tamper')


def test_numerical_work_rejects_login_nodes(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='compute allocation'):reference.require_compute()
    monkeypatch.setenv('SLURM_JOB_ID','unit-test')
    monkeypatch.setattr(reference.platform,'node',lambda:'arf-ui1.ib')
    with pytest.raises(RuntimeError,match='login node'):reference.require_compute()


def test_wall_times_excluded_but_accounting_preserved():
    assert reference.canonical({'elapsed_seconds':3,'timings':{'total_seconds':2},'calls':0})=={'calls':0}
    assert reference.canonical({'score':float('inf')})=={'score':{'nonfinite_diagnostic':'inf'}}


def test_fit_metadata_uses_the_production_record_serializer():
    from mutaco.surrogate import GPFitDiagnostic, GPFitStatus
    diagnostic=GPFitDiagnostic(GPFitStatus.SUCCESS,'fit','data','model',0.1,(),None,None,1,None,
                              {'status':'stopped'},None,None,(),{'fit_seed':7},{},())
    result=reference.fit_document(24,diagnostic)
    assert result['training_version']==24 and result['fit_id']=='fit'
    assert result['configuration']['fit_seed']==7


def test_library_enumeration_order_is_not_an_environment_change(tmp_path):
    first=fixture();second=deepcopy(first)
    first.pop('environment_signature_schema');second.pop('environment_signature_schema')
    first['environment_signature']=reference.environment_signature(first['environment'],legacy_ordered=True)
    second['environment']['runtime_threadpools'].reverse()
    second['environment_signature']=reference.environment_signature(second['environment'],legacy_ordered=True)
    a=tmp_path/'a.gz';b=tmp_path/'b.gz'
    reference.write_gzip(a,first);reference.write_gzip(b,second)
    report=reference.compare_files(a,b)
    assert report['artifacts_self_consistent'] and report['same_environment']
    assert report['runtime_enumeration_order_only_difference']
    for field,value in (('num_threads',1),('version','changed'),('filepath','other-library.so')):
        changed=deepcopy(second)
        changed['environment']['runtime_threadpools'][0][field]=value
        changed['environment_signature']=reference.environment_signature(changed['environment'],legacy_ordered=True)
        c=tmp_path/(field+'.gz');reference.write_gzip(c,changed)
        assert not reference.compare_files(a,c)['same_environment']
