"""The local CA and server certificate that Attendee's bot and the browser trust."""

import ssl
import stat
import subprocess
import sys

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import padding

from scripts.make_certs import make_certs
from tests.helpers import REPO


def handshake(cafile, certfile, keyfile, hostname):
    """A TLS handshake in memory; raises if the client doesn't trust the server for ``hostname``."""
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(certfile, keyfile)
    client_ctx = ssl.create_default_context(cafile=str(cafile))
    client_ctx.verify_flags |= ssl.VERIFY_X509_STRICT
    c_in, c_out, s_in, s_out = (ssl.MemoryBIO() for _ in range(4))
    client = client_ctx.wrap_bio(c_in, c_out, server_hostname=hostname)
    server = server_ctx.wrap_bio(s_in, s_out, server_side=True)
    done = {"client": False, "server": False}
    for _ in range(20):
        for name, side in (("client", client), ("server", server)):
            if not done[name]:
                try:
                    side.do_handshake()
                    done[name] = True
                except ssl.SSLWantReadError:
                    pass
        s_in.write(c_out.read())
        c_in.write(s_out.read())
        if all(done.values()):
            return
    raise AssertionError("handshake didn't finish")


def test_make_certs_san_and_chain(tmp_path):
    make_certs(tmp_path, ["interviewer", "localhost"])
    ca = x509.load_pem_x509_certificate((tmp_path / "ca.crt").read_bytes())
    leaf = x509.load_pem_x509_certificate((tmp_path / "server.crt").read_bytes())
    sans = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName)
    assert set(sans) == {"interviewer", "localhost"} and leaf.issuer == ca.subject
    ca.public_key().verify(leaf.signature, leaf.tbs_certificate_bytes, padding.PKCS1v15(), leaf.signature_hash_algorithm)
    assert (tmp_path / "ca.crt").read_text().strip() in (tmp_path / "ca-bundle.pem").read_text()


def test_certs_pass_a_strict_tls_handshake_for_each_host(tmp_path):
    make_certs(tmp_path, ["interviewer", "localhost"])
    for host in ("interviewer", "localhost"):
        handshake(tmp_path / "ca.crt", tmp_path / "server.crt", tmp_path / "server.key", host)
        handshake(tmp_path / "ca-bundle.pem", tmp_path / "server.crt", tmp_path / "server.key", host)


def test_keys_are_private(tmp_path):
    make_certs(tmp_path, ["localhost"])
    for name in ("ca.key", "server.key"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_cli_writes_into_out(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "make_certs.py"), "--out", str(tmp_path / "certs"),
                             "--hosts", "interviewer,localhost"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in (tmp_path / "certs").iterdir()} == {"ca.crt", "ca.key", "server.crt", "server.key",
                                                                 "ca-bundle.pem"}
