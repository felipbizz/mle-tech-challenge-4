from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["./settings.toml", "./.secrets.toml"],
)


# Exemplo de .secrets.toml
# minio_url = 'http://127.0.0.1:9000'
# access_key = 'acess_key-do-minio'
# secret_key = 'secret_key-do-minio'
# bucket_name = 'nome-do-bucket'
# path_to_table = 'nome-do-deltalake'
