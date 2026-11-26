"""Makes a local certificate authority and a server certificate for the interviewer.

Attendee's bot connects to wss://interviewer:8443 and your browser to
https://localhost:8443, so the server certificate names both. Attendee trusts
it through ca-bundle.pem (the usual public CAs plus ca.crt); Windows trusts it
once you import ca.crt (optional, see the README).

Usage: python scripts/make_certs.py --out certs --hosts interviewer,localhost
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
from pathlib import Path

import certifi
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _write(path: Path, data: bytes, private: bool = False) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o600 if private else 0o644)
    os.replace(tmp, path)


def _key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _san(hosts: list[str]) -> x509.SubjectAlternativeName:
    names: list[x509.GeneralName] = []
    for host in hosts:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    return x509.SubjectAlternativeName(names)


def make_certs(out_dir: Path, hosts: list[str], days: int = 825) -> None:
    """Write ca.crt, ca.key, server.crt, server.key and ca-bundle.pem into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC)
    start, end = now - dt.timedelta(minutes=5), now + dt.timedelta(days=days)

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AI Interviewer local CA")])
    ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(ca_ski, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=True,
                                     data_encipherment=False, key_agreement=False, key_cert_sign=False,
                                     crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(_san(hosts), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    _write(out / "ca.key", _key_pem(ca_key), private=True)
    _write(out / "ca.crt", ca_pem)
    _write(out / "server.key", _key_pem(key), private=True)
    _write(out / "server.crt", cert.public_bytes(serialization.Encoding.PEM))
    public = Path(certifi.where()).read_bytes()
    _write(out / "ca-bundle.pem", public.rstrip(b"\n") + b"\n\n" + ca_pem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make a local CA and a server certificate for the interviewer.")
    parser.add_argument("--out", default="certs", help="output folder (default: certs)")
    parser.add_argument("--hosts", default="interviewer,localhost",
                        help="comma-separated names the server certificate covers (default: interviewer,localhost)")
    parser.add_argument("--days", type=int, default=825, help="validity in days (default: 825)")
    args = parser.parse_args(argv)
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts:
        parser.error("--hosts needs at least one name")
    make_certs(Path(args.out), hosts, args.days)
    print(f"Wrote ca.crt, ca.key, server.crt, server.key and ca-bundle.pem to {args.out} for {', '.join(hosts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
