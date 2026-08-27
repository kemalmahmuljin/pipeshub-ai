"""Tests for ServiceNow connector."""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.config.constants.arangodb import Connectors, MimeTypes
from app.connectors.sources.servicenow.servicenow.constants import ORGANIZATIONAL_ENTITIES
from app.connectors.sources.servicenow.servicenow.connector import ServiceNowConnector
from app.models.entities import AppUser, AppUserGroup, FileRecord, RecordGroupType, RecordType, WebpageRecord
from app.models.permission import EntityType, Permission, PermissionType
from app.sources.client.servicenow.servicenow import ServiceNowRESTClientViaOAuthAuthorizationCode
from app.sources.external.servicenow.models import (
    ServiceNowAPIError,
    SysUserGroup,
    SysUserGroupMembership,
    SysUserRole,
    SysUserRoleAssignment,
    SysUserRoleContains,
    AttachmentMetadata,
    KBKnowledge,
    KBKnowledgeBase,
    KBCategory,
    RawPermission,
    SysUser,
    UserCriteria,
    TableAPIRecord,
    TableAPIResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_api_response(records: list) -> TableAPIResponse:
    """Match ServiceNowDataSource.get_now_table_tableName return type."""
    return TableAPIResponse(result=[TableAPIRecord(**r) for r in records])



def _kb_api_row(**fields: object) -> dict:
    base = {
        "description": "",
        "owner": None,
        "sys_created_on": "2024-01-01 00:00:00",
    }
    base.update(fields)
    return base


def _category_api_row(**fields: object) -> dict:
    base = {
        "value": "",
        "parent_id": None,
        "sys_created_on": "2024-01-01 00:00:00",
    }
    base.update(fields)
    return base


def _record_update(**kwargs):
    """Lightweight RecordUpdate stand-in for ServiceNow tests."""
    defaults = {
        "record": None,
        "is_new": True,
        "is_updated": False,
        "is_deleted": False,
        "metadata_changed": False,
        "content_changed": False,
        "permissions_changed": True,
        "new_permissions": [],
        "external_record_id": None,
    }
    defaults.update(kwargs)
    return type("RecordUpdate", (), defaults)()


def _sys_user_row(**fields: object) -> dict:
    """sys_user rows must include org reference keys so TableAPIRecord exposes .company, etc."""
    base = {
        "company": None,
        "department": None,
        "location": None,
        "cost_center": None,
    }
    base.update(fields)
    return base


def _make_mock_tx_store(existing_record=None, app_users=None):
    tx = AsyncMock()
    tx.get_record_by_external_id = AsyncMock(return_value=existing_record)
    tx.get_user_by_source_id = AsyncMock(return_value=None)
    tx.get_app_users = AsyncMock(return_value=app_users or [])
    tx.get_user_groups = AsyncMock(return_value=[])
    tx.create_user_group_membership = AsyncMock()
    tx.batch_upsert_user_groups = AsyncMock()
    tx.batch_upsert_record_groups = AsyncMock()
    tx.batch_upsert_record_group_permissions = AsyncMock()
    tx.get_record_group_by_external_id = AsyncMock(return_value=None)
    return tx


def _make_mock_data_store_provider(existing_record=None, app_users=None):
    tx = _make_mock_tx_store(existing_record, app_users)
    provider = MagicMock()

    @asynccontextmanager
    async def _transaction():
        yield tx

    provider.transaction = _transaction
    provider._tx_store = tx
    return provider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_logger():
    return logging.getLogger("test.servicenow")


@pytest.fixture()
def mock_data_entities_processor():
    class _Proc:
        org_id = "org-sn-1"

    proc = _Proc()
    proc.on_new_app_users = AsyncMock()
    proc.on_new_record_groups = AsyncMock()
    proc.on_new_records = AsyncMock()
    proc.on_new_user_groups = AsyncMock()
    proc.on_record_deleted = AsyncMock()
    proc.get_all_app_users = AsyncMock(return_value=[])
    proc.batch_upsert_user_groups = AsyncMock()
    proc.create_user_group_membership = AsyncMock()
    proc.get_user_by_source_id = AsyncMock(return_value=None)
    proc.get_record_by_external_id = AsyncMock(return_value=None)
    proc.get_records_by_parent = AsyncMock(return_value=[])
    proc.on_records_deleted_cascade = AsyncMock(return_value={"deleted_records": []})
    return proc


@pytest.fixture()
def mock_data_store_provider():
    return _make_mock_data_store_provider()


@pytest.fixture()
def mock_config_service():
    svc = AsyncMock()
    svc.get_config = AsyncMock(return_value={
        "auth": {
            "oauthConfigId": "oauth-sn-1",
        },
        "credentials": {
            "access_token": "sn-access-token",
            "refresh_token": "sn-refresh-token",
        },
    })
    return svc


@pytest.fixture()
def servicenow_connector(mock_logger, mock_data_entities_processor,
                          mock_data_store_provider, mock_config_service):
    with patch("app.connectors.sources.servicenow.servicenow.connector.ServicenowApp"):
        connector = ServiceNowConnector(
            logger=mock_logger,
            data_entities_processor=mock_data_entities_processor,
            data_store_provider=mock_data_store_provider,
            config_service=mock_config_service,
            connector_id="sn-conn-1",
            scope="personal",
            created_by="test-user-id",
        )
    connector.connector_name = Connectors.SERVICENOW
    return connector


# ===========================================================================
# Constants
# ===========================================================================

class TestOrganizationalEntities:
    def test_company_config(self):
        assert "company" in ORGANIZATIONAL_ENTITIES
        assert ORGANIZATIONAL_ENTITIES["company"]["table"] == "core_company"

    def test_department_config(self):
        assert "department" in ORGANIZATIONAL_ENTITIES
        assert ORGANIZATIONAL_ENTITIES["department"]["table"] == "cmn_department"

    def test_location_config(self):
        assert "location" in ORGANIZATIONAL_ENTITIES
        assert ORGANIZATIONAL_ENTITIES["location"]["table"] == "cmn_location"

    def test_cost_center_config(self):
        assert "cost_center" in ORGANIZATIONAL_ENTITIES
        assert ORGANIZATIONAL_ENTITIES["cost_center"]["table"] == "cmn_cost_center"

    def test_all_have_required_fields(self):
        for entity_type, config in ORGANIZATIONAL_ENTITIES.items():
            assert "table" in config
            assert "fields" in config
            assert "prefix" in config
            assert "sync_point_key" in config


# ===========================================================================
# ServiceNowConnector init
# ===========================================================================

class TestServiceNowConnectorInit:
    def test_constructor(self, servicenow_connector):
        assert servicenow_connector.connector_id == "sn-conn-1"
        assert servicenow_connector.servicenow_client is None
        assert servicenow_connector.servicenow_datasource is None
        assert servicenow_connector.instance_url is None

    def test_sync_points_created(self, servicenow_connector):
        assert servicenow_connector.user_sync_point is not None
        assert servicenow_connector.category_sync_point is not None
        assert servicenow_connector.article_sync_point is not None

    def test_no_sync_point_for_groups_roles_and_knowledge_bases(self, servicenow_connector):
        """These three reads must stay full, so none may carry a checkpoint."""
        assert not hasattr(servicenow_connector, "group_sync_point")
        assert not hasattr(servicenow_connector, "role_assignment_sync_point")
        assert not hasattr(servicenow_connector, "kb_sync_point")

    def test_org_entity_sync_points(self, servicenow_connector):
        for key in ["company", "department", "location", "cost_center"]:
            assert key in servicenow_connector.org_entity_sync_points

    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowRESTClientViaOAuthAuthorizationCode")
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowDataSource")
    async def test_init_success(self, mock_ds_cls, mock_client_cls, mock_fetch_oauth,
                                servicenow_connector):
        mock_fetch_oauth.return_value = {
            "config": {
                "clientId": "sn-client-id",
                "clientSecret": "sn-client-secret",
                "instanceUrl": "https://dev12345.service-now.com",
            },
            "redirectUri": "http://localhost/callback",
        }
        mock_client_cls.return_value = MagicMock()
        mock_ds_instance = MagicMock()
        mock_ds_instance.get_now_table_tableName = AsyncMock(
            return_value=_table_api_response([])
        )
        mock_ds_cls.return_value = mock_ds_instance

        result = await servicenow_connector.init()
        assert result is True
        assert servicenow_connector.instance_url == "https://dev12345.service-now.com"

    async def test_init_fails_no_config(self, servicenow_connector):
        servicenow_connector.config_service.get_config = AsyncMock(return_value=None)
        assert await servicenow_connector.init() is False

    async def test_init_fails_no_oauth_config_id(self, servicenow_connector):
        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "auth": {},
            "credentials": {},
        })
        assert await servicenow_connector.init() is False

    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    async def test_init_fails_oauth_not_found(self, mock_fetch_oauth, servicenow_connector):
        mock_fetch_oauth.return_value = None
        assert await servicenow_connector.init() is False

    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    async def test_init_fails_incomplete_config(self, mock_fetch_oauth, servicenow_connector):
        mock_fetch_oauth.return_value = {"config": {"clientId": "id"}}
        assert await servicenow_connector.init() is False

    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    async def test_init_fails_no_access_token(self, mock_fetch_oauth, servicenow_connector):
        mock_fetch_oauth.return_value = {
            "config": {
                "clientId": "id", "clientSecret": "secret",
                "instanceUrl": "https://sn.example.com",
                "redirectUri": "http://localhost/callback",
            }
        }
        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "auth": {"oauthConfigId": "oauth-sn-1"},
            "credentials": {},
        })
        assert await servicenow_connector.init() is False

    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowRESTClientViaOAuthAuthorizationCode")
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowDataSource")
    async def test_init_fails_connection_test(self, mock_ds_cls, mock_client_cls, mock_fetch_oauth,
                                              servicenow_connector):
        mock_fetch_oauth.return_value = {
            "config": {
                "clientId": "id", "clientSecret": "secret",
                "instanceUrl": "https://sn.example.com",
                "redirectUri": "http://localhost/callback",
            }
        }
        mock_client_cls.return_value = MagicMock()
        mock_ds_instance = MagicMock()
        mock_ds_instance.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(401, "Unauthorized", None)
        )
        mock_ds_cls.return_value = mock_ds_instance
        assert await servicenow_connector.init() is False


# ===========================================================================
# _get_fresh_datasource
# ===========================================================================

class TestGetFreshDatasource:
    async def test_raises_when_client_not_initialized(self, servicenow_connector):
        with pytest.raises(Exception, match="not initialized"):
            await servicenow_connector._get_fresh_datasource()

    async def test_returns_datasource_with_fresh_token(self, servicenow_connector):
        """A real client, because the transport sends headers, not the attribute."""
        client = ServiceNowRESTClientViaOAuthAuthorizationCode(
            instance_url="https://dev194883.service-now.com",
            client_id="cid", client_secret="secret",
            redirect_uri="http://localhost/cb", access_token="old-token",
        )
        servicenow_connector.servicenow_client = client
        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "credentials": {"access_token": "new-token"},
        })
        ds = await servicenow_connector._get_fresh_datasource()
        assert ds is not None
        assert client.access_token == "new-token"
        assert client.headers["Authorization"] == "Bearer new-token"

    async def test_no_config_raises(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        servicenow_connector.config_service.get_config = AsyncMock(return_value=None)
        with pytest.raises(Exception, match="not found"):
            await servicenow_connector._get_fresh_datasource()

    async def test_no_token_raises(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "credentials": {},
        })
        with pytest.raises(Exception, match="No access token"):
            await servicenow_connector._get_fresh_datasource()

    async def test_same_token_no_update(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        servicenow_connector.servicenow_client.access_token = "same-token"
        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "credentials": {"access_token": "same-token"},
        })
        ds = await servicenow_connector._get_fresh_datasource()
        assert ds is not None


# ===========================================================================
# test_connection_and_access
# ===========================================================================

class TestConnectionAndAccess:
    async def test_success(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        servicenow_connector.servicenow_client.access_token = "token"
        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "credentials": {"access_token": "token"},
        })
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([{"sys_id": "1"}])
            )
            mock_ds.return_value = mock_datasource
            assert await servicenow_connector.test_connection_and_access() is True

    async def test_failure(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                side_effect=ServiceNowAPIError(401, "Unauthorized", None)
            )
            mock_ds.return_value = mock_datasource
            assert await servicenow_connector.test_connection_and_access() is False

    async def test_exception(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock,
                          side_effect=Exception("Connection refused")):
            assert await servicenow_connector.test_connection_and_access() is False


# ===========================================================================
# stream_record
# ===========================================================================

class TestStreamRecord:
    async def test_stream_article(self, servicenow_connector):
        record = MagicMock()
        record.record_type = RecordType.WEBPAGE
        record.record_name = "Article 1"
        record.external_record_id = "art-1"

        with patch.object(servicenow_connector, "_fetch_article_content",
                          new_callable=AsyncMock, return_value="<h1>Hello</h1>"):
            response = await servicenow_connector.stream_record(record)
            assert response is not None

    async def test_stream_attachment(self, servicenow_connector):
        record = MagicMock()
        record.record_type = RecordType.FILE
        record.record_name = "file.pdf"
        record.external_record_id = "att-1"
        record.id = "rec-1"
        record.mime_type = "application/pdf"

        with patch.object(servicenow_connector, "_fetch_attachment_content",
                          new_callable=AsyncMock, return_value=b"PDF content"):
            response = await servicenow_connector.stream_record(record)
            assert response is not None

    async def test_unsupported_type_raises(self, servicenow_connector):
        record = MagicMock()
        record.record_type = RecordType.MAIL
        record.record_name = "email"
        record.external_record_id = "mail-1"

        with pytest.raises(HTTPException) as exc_info:
            await servicenow_connector.stream_record(record)
        assert exc_info.value.status_code == 400

    async def test_stream_exception_raises_500(self, servicenow_connector):
        record = MagicMock()
        record.record_type = RecordType.WEBPAGE
        record.record_name = "Article"
        record.external_record_id = "art-1"

        with patch.object(servicenow_connector, "_fetch_article_content",
                          new_callable=AsyncMock, side_effect=Exception("Network error")):
            with pytest.raises(HTTPException) as exc_info:
                await servicenow_connector.stream_record(record)
            assert exc_info.value.status_code == 500


# ===========================================================================
# _fetch_article_content
# ===========================================================================

class TestFetchArticleContent:
    async def test_success(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "art-1", "text": "<p>Content</p>", "number": "KB001"},
                ])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_article_content("art-1")
            assert result == "<p>Content</p>"

    async def test_not_found(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            with pytest.raises(HTTPException) as exc_info:
                await servicenow_connector._fetch_article_content("nonexistent")
            assert exc_info.value.status_code == 404

    async def test_empty_content(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "art-1", "text": "", "number": "KB001"},
                ])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_article_content("art-1")
            assert result == "<p>No content available</p>"

    async def test_api_failure(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                side_effect=ServiceNowAPIError(404, "Server error", None)
            )
            mock_ds.return_value = mock_datasource
            with pytest.raises(HTTPException) as exc_info:
                await servicenow_connector._fetch_article_content("art-1")
            assert exc_info.value.status_code == 404


# ===========================================================================
# _fetch_attachment_content
# ===========================================================================

    @pytest.mark.asyncio
    async def test_fetch_article_content_generic_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("ds fail"))
        with pytest.raises(HTTPException) as exc_info:
            await servicenow_connector._fetch_article_content("art-1")
        assert exc_info.value.status_code == 500


class TestFetchAttachmentContent:
    async def test_success(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.download_attachment = AsyncMock(return_value=b"file content")
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_attachment_content("att-1")
            assert result == b"file content"

    async def test_not_found(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.download_attachment = AsyncMock(return_value=None)
            mock_ds.return_value = mock_datasource
            with pytest.raises(HTTPException) as exc_info:
                await servicenow_connector._fetch_attachment_content("nonexistent")
            assert exc_info.value.status_code == 404

    async def test_exception(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.download_attachment = AsyncMock(side_effect=Exception("Download failed"))
            mock_ds.return_value = mock_datasource
            with pytest.raises(HTTPException) as exc_info:
                await servicenow_connector._fetch_attachment_content("att-1")
            assert exc_info.value.status_code == 500


# ===========================================================================
# get_signed_url, handle_webhook, cleanup, reindex, get_filter_options
# ===========================================================================

    @pytest.mark.asyncio
    async def test_fetch_attachment_content_api_error(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.download_attachment = AsyncMock(
            side_effect=ServiceNowAPIError(503, "unavailable", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)
        with pytest.raises(HTTPException) as exc_info:
            await servicenow_connector._fetch_attachment_content("att-1")
        assert exc_info.value.status_code == 503


class TestMiscMethods:
    def test_get_signed_url_returns_none(self, servicenow_connector):
        assert servicenow_connector.get_signed_url(MagicMock()) is None

    async def test_handle_webhook_returns_true(self, servicenow_connector):
        result = await servicenow_connector.handle_webhook_notification("org-1", {"event": "test"})
        assert result is True

    async def test_cleanup(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        servicenow_connector.servicenow_datasource = MagicMock()
        await servicenow_connector.cleanup()
        assert servicenow_connector.servicenow_client is None
        assert servicenow_connector.servicenow_datasource is None

    async def test_reindex_records(self, servicenow_connector):
        await servicenow_connector.reindex_records([MagicMock()])
        # No-op, just verify it doesn't raise

    async def test_get_filter_options_raises(self, servicenow_connector):
        with pytest.raises(NotImplementedError):
            await servicenow_connector.get_filter_options("key")

    async def test_run_incremental_sync_delegates(self, servicenow_connector):
        with patch.object(servicenow_connector, "run_sync", new_callable=AsyncMock) as mock_sync:
            await servicenow_connector.run_incremental_sync()
            mock_sync.assert_called_once()


# ===========================================================================
# _get_admin_users
# ===========================================================================

class TestGetAdminUsers:
    async def test_finds_admin_users(self, servicenow_connector):
        mock_app_user = MagicMock(spec=AppUser)
        mock_app_user.email = "admin@example.com"

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([{"user": "sys-admin-1"}])
            )
            mock_ds.return_value = mock_datasource

            servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(return_value=mock_app_user)

            result = await servicenow_connector._get_admin_users()
            assert len(result) == 1
            assert result[0].email == "admin@example.com"

    async def test_no_admin_users_found(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                side_effect=ServiceNowAPIError(404, "Not found", None)
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._get_admin_users()
            assert result == []

    async def test_dict_reference_field(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([{"user": "sys-admin-1"}])
            )
            mock_ds.return_value = mock_datasource

            tx = _make_mock_tx_store()
            tx.get_user_by_source_id = AsyncMock(return_value=None)

            @asynccontextmanager
            async def _tx():
                yield tx

            servicenow_connector.data_store_provider = MagicMock()
            servicenow_connector.data_store_provider.transaction = _tx

            result = await servicenow_connector._get_admin_users()
            assert result == []

    async def test_exception_returns_empty(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock,
                          side_effect=Exception("Network error")):
            result = await servicenow_connector._get_admin_users()
            assert result == []


# ===========================================================================
# _fetch_all_groups
# ===========================================================================

class TestFetchAllGroups:
    async def test_fetches_groups(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "g1", "name": "Group 1"},
                    {"sys_id": "g2", "name": "Group 2"},
                ])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_groups()
            assert len(result) == 2

    async def test_empty_results(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_groups()
            assert result == []

    async def test_api_failure_propagates(self, servicenow_connector):
        """A partial group list would make the caller delete edges it cannot rebuild."""
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                side_effect=ServiceNowAPIError(500, "Error", None)
            )
            mock_ds.return_value = mock_datasource
            with pytest.raises(ServiceNowAPIError):
                await servicenow_connector._fetch_all_groups()


# ===========================================================================
# _fetch_all_memberships
# ===========================================================================

class TestFetchAllMemberships:
    @pytest.mark.asyncio
    async def test_membership_query_carries_no_delta_even_with_a_sync_point(self, servicenow_connector):
        """on_new_user_groups replaces every edge, so a delta read loses members."""
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value={"last_sync_time": "2026-08-26 09:00:00"})
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.group_sync_point = sync_point

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._fetch_all_memberships()

        query = mock_ds.get_now_table_tableName.call_args.kwargs["sysparm_query"]
        assert "sys_updated_on>" not in query

    async def test_fetches_memberships(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "m1", "user": "u1", "group": "g1", "sys_updated_on": "2024-01-01"},
                ])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_memberships()
            assert len(result) == 1

    async def test_reads_every_row(self, servicenow_connector):
        """A delta read plus a replace write removes members that nobody removed."""
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_memberships()
            assert result == []
            call_kwargs = mock_datasource.get_now_table_tableName.call_args.kwargs
            assert "sys_updated_on>" not in call_kwargs["sysparm_query"]


# ===========================================================================
# _flatten_and_create_user_groups
# ===========================================================================

class TestFlattenAndCreateUserGroups:
    async def test_simple_flatten(self, servicenow_connector):
        groups = [
            SysUserGroup(sys_id="g1", name="Group 1"),
            SysUserGroup(sys_id="g2", name="Group 2", parent="g1"),
        ]
        memberships = [
            SysUserGroupMembership(sys_id="m1", user="u1", group="g1"),
            SysUserGroupMembership(sys_id="m2", user="u2", group="g2"),
        ]

        mock_user1 = MagicMock(spec=AppUser)
        mock_user1.source_user_id = "u1"
        mock_user2 = MagicMock(spec=AppUser)
        mock_user2.source_user_id = "u2"

        servicenow_connector.data_entities_processor.get_all_app_users = AsyncMock(return_value=[mock_user1, mock_user2])

        with patch.object(servicenow_connector, "_transform_to_user_group") as mock_transform:
            mock_group = MagicMock(spec=AppUserGroup)
            mock_group.name = "Group"
            mock_transform.return_value = mock_group

            result = await servicenow_connector._flatten_and_create_user_groups(groups, memberships)
            assert len(result) == 2
            # Group g1 should have users from g1 + children (g2)
            g1_result = [r for r in result if True]  # All results
            assert len(g1_result) == 2

    async def test_string_references(self, servicenow_connector):
        """Test with string references instead of dict references."""
        groups = [SysUserGroup(sys_id="g1", name="Group 1")]
        memberships = [SysUserGroupMembership(sys_id="m1", user="u1", group="g1")]

        servicenow_connector.data_entities_processor.get_all_app_users = AsyncMock(return_value=[])

        with patch.object(servicenow_connector, "_transform_to_user_group") as mock_transform:
            mock_group = MagicMock(spec=AppUserGroup)
            mock_group.name = "Group"
            mock_transform.return_value = mock_group

            result = await servicenow_connector._flatten_and_create_user_groups(groups, memberships)
            assert len(result) == 1


# ===========================================================================
# _fetch_all_roles and _fetch_all_role_assignments
# ===========================================================================

    @pytest.mark.asyncio
    async def test_flatten_skips_none_user_group(self, servicenow_connector):
        groups = [SysUserGroup(sys_id="g1", name="Group")]
        memberships = [SysUserGroupMembership(sys_id="m1", user="u1", group="g1")]
        servicenow_connector.data_entities_processor.get_all_app_users = AsyncMock(return_value=[])

        with patch.object(servicenow_connector, "_transform_to_user_group", return_value=None):
            result = await servicenow_connector._flatten_and_create_user_groups(groups, memberships)
        assert result == []

    @pytest.mark.asyncio
    async def test_flatten_exception_propagates(self, servicenow_connector):
        servicenow_connector.data_entities_processor.get_all_app_users = AsyncMock(side_effect=RuntimeError("tx fail"))
        with pytest.raises(RuntimeError, match="tx fail"):
            await servicenow_connector._flatten_and_create_user_groups([], [])


class TestFetchRoles:
    async def test_fetches_roles(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([{"sys_id": "r1", "name": "admin"}])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_roles()
            assert len(result) == 1

    async def test_fetches_role_assignments(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "ra1", "user": "u1", "role": "r1", "sys_updated_on": "2024-01-01"},
                ])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_role_assignments()
            assert len(result) == 1
            assert result[0].role == "r1"
            assert result[0].user == "u1"

    async def test_fetches_role_hierarchy(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "h1", "contains": "r1", "role": "r2"},
                ])
            )
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_role_hierarchy()
            assert len(result) == 1


# ===========================================================================
# _sync_users
# ===========================================================================

    @pytest.mark.asyncio
    async def test_fetch_all_roles_empty_page_breaks(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)
        assert await servicenow_connector._fetch_all_roles() == []

    @pytest.mark.asyncio
    async def test_fetch_all_roles_outer_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("roles fetch fail"))
        with pytest.raises(RuntimeError, match="roles fetch fail"):
            await servicenow_connector._fetch_all_roles()

    @pytest.mark.asyncio
    async def test_fetch_all_role_assignments_reads_every_row(self, servicenow_connector):
        """A delta read plus a replace write removes roles that nobody revoked."""
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._fetch_all_role_assignments()
        call_kwargs = mock_ds.get_now_table_tableName.call_args.kwargs
        assert "sys_updated_on>" not in call_kwargs["sysparm_query"]

    @pytest.mark.asyncio
    async def test_fetch_all_role_assignments_outer_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(
            side_effect=RuntimeError("datasource fail")
        )
        with pytest.raises(RuntimeError, match="datasource fail"):
            await servicenow_connector._fetch_all_role_assignments()

    @pytest.mark.asyncio
    async def test_fetch_role_hierarchy_outer_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("hierarchy fail"))
        with pytest.raises(RuntimeError, match="hierarchy fail"):
            await servicenow_connector._fetch_role_hierarchy()


class TestSyncUsers:
    async def test_syncs_users(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.user_sync_point.update_sync_point = AsyncMock()

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_app_user", new_callable=AsyncMock) as mock_transform:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    _sys_user_row(
                        sys_id="u1",
                        user_name="user1",
                        email="user1@example.com",
                        first_name="User",
                        last_name="One",
                        active="true",
                        sys_updated_on="2024-01-01",
                    ),
                ])
            )
            mock_ds.return_value = mock_datasource

            mock_app_user = MagicMock(spec=AppUser)
            mock_transform.return_value = mock_app_user

            await servicenow_connector._sync_users()
            servicenow_connector.data_entities_processor.on_new_app_users.assert_called_once()

    async def test_skips_users_without_email(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.user_sync_point.update_sync_point = AsyncMock()

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "u1", "email": "", "sys_updated_on": "2024-01-01"},
                ])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_users()
            servicenow_connector.data_entities_processor.on_new_app_users.assert_not_called()


# ===========================================================================
# run_sync
# ===========================================================================

class TestRunSync:
    async def test_raises_when_client_not_initialized(self, servicenow_connector):
        with pytest.raises(Exception, match="not initialized"):
            await servicenow_connector.run_sync()

    async def test_full_sync_flow(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        with patch.object(servicenow_connector, "_sync_users_and_groups", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_get_admin_users", new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_sync_knowledge_bases", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_sync_categories", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_sync_articles", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_remove_deleted_record_groups", new_callable=AsyncMock):
            await servicenow_connector.run_sync()

    async def test_sync_continues_without_admin_users(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        with patch.object(servicenow_connector, "_sync_users_and_groups", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_get_admin_users", new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_sync_knowledge_bases", new_callable=AsyncMock) as mock_kb, \
             patch.object(servicenow_connector, "_sync_categories", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_sync_articles", new_callable=AsyncMock), \
             patch.object(servicenow_connector, "_remove_deleted_record_groups", new_callable=AsyncMock):
            await servicenow_connector.run_sync()
            mock_kb.assert_called_once_with([])

    async def test_sync_propagates_exceptions(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        with patch.object(servicenow_connector, "_sync_users_and_groups", new_callable=AsyncMock,
                          side_effect=Exception("sync error")):
            with pytest.raises(Exception, match="sync error"):
                await servicenow_connector.run_sync()


# ===========================================================================
# Deep sync: _sync_users_and_groups
# ===========================================================================

class TestSyncUsersAndGroups:
    async def test_calls_all_sub_methods(self, servicenow_connector):
        with patch.object(servicenow_connector, "_sync_organizational_entities", new_callable=AsyncMock) as mock_oe, \
             patch.object(servicenow_connector, "_sync_users", new_callable=AsyncMock) as mock_u, \
             patch.object(servicenow_connector, "_sync_user_groups", new_callable=AsyncMock) as mock_g, \
             patch.object(servicenow_connector, "_sync_roles", new_callable=AsyncMock) as mock_r:
            await servicenow_connector._sync_users_and_groups()
            mock_oe.assert_called_once()
            mock_u.assert_called_once()
            mock_g.assert_called_once()
            mock_r.assert_called_once()

    async def test_propagates_exception(self, servicenow_connector):
        with patch.object(servicenow_connector, "_sync_organizational_entities",
                          new_callable=AsyncMock, side_effect=Exception("org fail")):
            with pytest.raises(Exception, match="org fail"):
                await servicenow_connector._sync_users_and_groups()


# ===========================================================================
# Deep sync: _sync_user_groups
# ===========================================================================

class TestSyncUserGroups:
    async def test_skips_when_no_memberships(self, servicenow_connector):
        with patch.object(servicenow_connector, "_fetch_all_memberships",
                          new_callable=AsyncMock, return_value=[]):
            await servicenow_connector._sync_user_groups()
            servicenow_connector.data_entities_processor.on_new_user_groups.assert_not_called()

    async def test_processes_groups_and_memberships(self, servicenow_connector):
        memberships = [{"user": "u1", "group": "g1"}]
        groups = [{"sys_id": "g1", "name": "Group 1"}]
        mock_result = [(MagicMock(), [MagicMock()])]
        with patch.object(servicenow_connector, "_fetch_all_memberships",
                          new_callable=AsyncMock, return_value=memberships), \
             patch.object(servicenow_connector, "_fetch_all_groups",
                          new_callable=AsyncMock, return_value=groups), \
             patch.object(servicenow_connector, "_flatten_and_create_user_groups",
                          new_callable=AsyncMock, return_value=mock_result):
            await servicenow_connector._sync_user_groups()
            servicenow_connector.data_entities_processor.on_new_user_groups.assert_called_once()


# ===========================================================================
# Deep sync: _sync_roles
# ===========================================================================

    @pytest.mark.asyncio
    async def test_sync_user_groups_exception_propagates(self, servicenow_connector):
        with patch.object(
            servicenow_connector, "_fetch_all_memberships",
            new_callable=AsyncMock, side_effect=RuntimeError("membership fail"),
        ):
            with pytest.raises(RuntimeError, match="membership fail"):
                await servicenow_connector._sync_user_groups()


class TestSyncRoles:
    async def test_skips_when_no_role_assignments(self, servicenow_connector):
        with patch.object(servicenow_connector, "_fetch_all_role_assignments",
                          new_callable=AsyncMock, return_value=[]):
            await servicenow_connector._sync_roles()
            servicenow_connector.data_entities_processor.on_new_user_groups.assert_not_called()

    async def test_adds_role_prefix(self, servicenow_connector):
        assignments = [
            SysUserRoleAssignment(
                sys_id="ra1",
                user="u1",
                role="r1",
                sys_updated_on="2024-01-01",
            ),
        ]
        roles = [SysUserRole(sys_id="r1", name="admin")]
        hierarchy = []
        mock_group = MagicMock(spec=AppUserGroup)
        mock_group.name = "admin"
        mock_result = [(mock_group, [MagicMock()])]
        with patch.object(servicenow_connector, "_fetch_all_role_assignments",
                          new_callable=AsyncMock, return_value=assignments), \
             patch.object(servicenow_connector, "_fetch_all_roles",
                          new_callable=AsyncMock, return_value=roles), \
             patch.object(servicenow_connector, "_fetch_role_hierarchy",
                          new_callable=AsyncMock, return_value=hierarchy), \
             patch.object(servicenow_connector, "_flatten_and_create_user_groups",
                          new_callable=AsyncMock, return_value=mock_result):
            await servicenow_connector._sync_roles()
            assert mock_group.name.startswith("ROLE_")


# ===========================================================================
# Deep sync: _sync_organizational_entities
# ===========================================================================

    @pytest.mark.asyncio
    async def test_sync_roles_with_hierarchy_merges_parent(self, servicenow_connector, mock_data_entities_processor):
        assignments = [
            SysUserRoleAssignment(sys_id="ra1", user="u1", role="child", sys_updated_on="2024-01-01"),
        ]
        roles = [
            SysUserRole(sys_id="parent", name="ParentRole"),
            SysUserRole(sys_id="child", name="ChildRole"),
        ]
        hierarchy = [SysUserRoleContains(sys_id="h1", role="child", contains="parent")]
        mock_group = MagicMock(spec=AppUserGroup)
        mock_group.name = "ChildRole"

        with patch.object(servicenow_connector, "_fetch_all_role_assignments", new_callable=AsyncMock, return_value=assignments), \
             patch.object(servicenow_connector, "_fetch_all_roles", new_callable=AsyncMock, return_value=roles), \
             patch.object(servicenow_connector, "_fetch_role_hierarchy", new_callable=AsyncMock, return_value=hierarchy), \
             patch.object(servicenow_connector, "_flatten_and_create_user_groups",
                          new_callable=AsyncMock, return_value=[(mock_group, [])]):
            await servicenow_connector._sync_roles()

        mock_data_entities_processor.on_new_user_groups.assert_called_once()
        assert mock_group.name.startswith("ROLE_")

    @pytest.mark.asyncio
    async def test_sync_roles_exception_propagates(self, servicenow_connector):
        with patch.object(
            servicenow_connector, "_fetch_all_role_assignments",
            new_callable=AsyncMock, side_effect=RuntimeError("roles fail"),
        ):
            with pytest.raises(RuntimeError, match="roles fail"):
                await servicenow_connector._sync_roles()


class TestSyncOrganizationalEntities:
    async def test_calls_sync_for_each_entity_type(self, servicenow_connector):
        with patch.object(servicenow_connector, "_sync_single_organizational_entity",
                          new_callable=AsyncMock) as mock_sync:
            await servicenow_connector._sync_organizational_entities()
            assert mock_sync.call_count == len(ORGANIZATIONAL_ENTITIES)

    async def test_propagates_exception(self, servicenow_connector):
        with patch.object(servicenow_connector, "_sync_single_organizational_entity",
                          new_callable=AsyncMock, side_effect=Exception("entity fail")):
            with pytest.raises(Exception, match="entity fail"):
                await servicenow_connector._sync_organizational_entities()


# ===========================================================================
# Deep sync: _sync_single_organizational_entity
# ===========================================================================

class TestSyncSingleOrganizationalEntity:
    async def test_full_sync_entities(self, servicenow_connector):
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value=None)
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.org_entity_sync_points = {"company": sync_point}

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_organizational_group", return_value=MagicMock()):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "c1", "name": "Company 1", "sys_updated_on": "2024-01-01"},
                ])
            )
            mock_ds.return_value = mock_datasource
            config = ORGANIZATIONAL_ENTITIES["company"]
            await servicenow_connector._sync_single_organizational_entity("company", config)
            sync_point.update_sync_point.assert_called_once()

    async def test_delta_sync_entities(self, servicenow_connector):
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value={"last_sync_time": "2024-01-01"})
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.org_entity_sync_points = {"department": sync_point}

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            config = ORGANIZATIONAL_ENTITIES["department"]
            await servicenow_connector._sync_single_organizational_entity("department", config)

    async def test_paginates_entities(self, servicenow_connector):
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value=None)
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.org_entity_sync_points = {"location": sync_point}

        page1_data = [{"sys_id": f"l{i}", "name": f"Location {i}", "sys_updated_on": "2024-01-01"} for i in range(100)]
        page2_data = [{"sys_id": "l100", "name": "Location 100", "sys_updated_on": "2024-01-02"}]

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_organizational_group", return_value=MagicMock()):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(side_effect=[
                _table_api_response(page1_data),
                _table_api_response(page2_data),
            ])
            mock_ds.return_value = mock_datasource
            config = ORGANIZATIONAL_ENTITIES["location"]
            await servicenow_connector._sync_single_organizational_entity("location", config)
            assert mock_datasource.get_now_table_tableName.call_count == 2

    @pytest.mark.asyncio
    async def test_api_error_logs_and_stops(self, servicenow_connector):
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value=None)
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.org_entity_sync_points = {"company": sync_point}

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(502, "bad gateway", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with patch.object(servicenow_connector.logger, "error") as mock_error, \
             pytest.raises(ServiceNowAPIError):
            await servicenow_connector._sync_single_organizational_entity(
                "company", ORGANIZATIONAL_ENTITIES["company"]
            )
        mock_error.assert_called()
        # A page that failed must not leave a checkpoint claiming it was read.
        sync_point.update_sync_point.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_point_read_exception_propagates(self, servicenow_connector):
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(side_effect=RuntimeError("sync fail"))
        servicenow_connector.org_entity_sync_points = {"company": sync_point}

        with pytest.raises(RuntimeError, match="sync fail"):
            await servicenow_connector._sync_single_organizational_entity(
                "company", ORGANIZATIONAL_ENTITIES["company"]
            )


# ===========================================================================
# Deep sync: _sync_knowledge_bases
# ===========================================================================

class TestSyncKnowledgeBases:
    @pytest.mark.asyncio
    async def test_knowledge_base_query_carries_no_delta_even_with_a_sync_point(self, servicenow_connector):
        """Read grants live in kb_uc_can_read_mtom, which never bumps the base row."""
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value={"last_sync_time": "2026-08-25 07:30:06"})
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.kb_sync_point = sync_point

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_knowledge_bases([])

        query = mock_ds.get_now_table_tableName.call_args.kwargs["sysparm_query"]
        assert "sys_updated_on>" not in query

    async def test_syncs_knowledge_bases(self, servicenow_connector):
        tx = _make_mock_tx_store()
        tx.get_record_group_by_external_id = AsyncMock(return_value=None)
        tx.batch_upsert_record_groups = AsyncMock()
        tx.batch_upsert_record_group_permissions = AsyncMock()

        @asynccontextmanager
        async def _tx():
            yield tx

        servicenow_connector.data_store_provider = MagicMock()
        servicenow_connector.data_store_provider.transaction = _tx

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_kb_record_group", return_value=MagicMock(id="rg-1")), \
             patch.object(servicenow_connector, "_fetch_kb_permissions_from_criteria", new_callable=AsyncMock,
                          return_value={"read": [], "write": []}), \
             patch.object(servicenow_connector, "_process_criteria_permissions", new_callable=AsyncMock,
                          return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects", new_callable=AsyncMock,
                          return_value=[]):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "kb1", "title": "KB 1", "owner": "o1", "sys_updated_on": "2024-01-01"},
                ])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_knowledge_bases([])
            call_kwargs = mock_datasource.get_now_table_tableName.call_args.kwargs
            assert "sys_updated_on>" not in call_kwargs["sysparm_query"]

    async def test_adds_admin_permissions(self, servicenow_connector):
        tx = _make_mock_tx_store()
        tx.get_record_group_by_external_id = AsyncMock(return_value=None)
        tx.batch_upsert_record_groups = AsyncMock()
        tx.batch_upsert_record_group_permissions = AsyncMock()

        @asynccontextmanager
        async def _tx():
            yield tx

        servicenow_connector.data_store_provider = MagicMock()
        servicenow_connector.data_store_provider.transaction = _tx

        admin_user = MagicMock(spec=AppUser)
        admin_user.email = "admin@example.com"

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_kb_record_group", return_value=MagicMock(id="rg-1")), \
             patch.object(servicenow_connector, "_fetch_kb_permissions_from_criteria", new_callable=AsyncMock,
                          return_value={"read": [], "write": []}), \
             patch.object(servicenow_connector, "_process_criteria_permissions", new_callable=AsyncMock,
                          return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects", new_callable=AsyncMock,
                          return_value=[]):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {"sys_id": "kb1", "title": "KB 1", "owner": None, "sys_updated_on": "2024-01-01"},
                ])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_knowledge_bases([admin_user])
            servicenow_connector.data_entities_processor.on_new_record_groups.assert_called()

    async def test_empty_kbs(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_knowledge_bases([])


# ===========================================================================
# Deep sync: _sync_users pagination
# ===========================================================================

class TestSyncUsersDeep:
    async def test_paginates_users(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.user_sync_point.update_sync_point = AsyncMock()

        page1 = [
            _sys_user_row(
                sys_id=f"u{i}",
                email=f"user{i}@example.com",
                sys_updated_on="2024-01-01",
            )
            for i in range(100)
        ]
        page2 = [_sys_user_row(sys_id="u100", email="user100@example.com", sys_updated_on="2024-01-02")]

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_app_user", new_callable=AsyncMock, return_value=MagicMock(spec=AppUser)):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(side_effect=[
                _table_api_response(page1),
                _table_api_response(page2),
            ])
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_users()
            assert servicenow_connector.data_entities_processor.on_new_app_users.call_count == 2

    async def test_delta_sync_users(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(
            return_value={"last_sync_time": "2024-01-01 00:00:00"}
        )
        servicenow_connector.user_sync_point.update_sync_point = AsyncMock()

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_app_user", new_callable=AsyncMock, return_value=MagicMock(spec=AppUser)):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    _sys_user_row(
                        sys_id="u1",
                        email="user@example.com",
                        sys_updated_on="2024-06-01",
                    ),
                ])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_users()
            servicenow_connector.user_sync_point.update_sync_point.assert_called()

    async def test_creates_org_entity_links(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.user_sync_point.update_sync_point = AsyncMock()

        user_data = {
            "sys_id": "u1",
            "email": "user@example.com",
            "company": "comp1",
            "department": "dept1",
            "location": "",
            "cost_center": None,
            "sys_updated_on": "2024-01-01",
        }

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_app_user", new_callable=AsyncMock, return_value=MagicMock(spec=AppUser)):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([user_data])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_users()
            # Should create links for company and department (not location/cost_center since empty)
            assert servicenow_connector.data_entities_processor.create_user_group_membership.call_count == 2


# ===========================================================================
# Deep sync: _sync_categories
# ===========================================================================

class TestSyncCategories:
    async def test_syncs_categories(self, servicenow_connector):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_category_record_group", return_value=MagicMock()):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {
                        "sys_id": "cat1",
                        "label": "Category 1",
                        "parent_table": None,
                        "parent_id": None,
                        "sys_updated_on": "2024-01-01",
                    },
                ])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_categories()
            servicenow_connector.category_sync_point.update_sync_point.assert_called()
            servicenow_connector.data_entities_processor.on_new_record_groups.assert_called()

    async def test_empty_categories(self, servicenow_connector):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_categories()

    async def test_categories_with_parent(self, servicenow_connector):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        mock_rg = MagicMock()
        mock_rg.parent_external_group_id = "cat1"
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds, \
             patch.object(servicenow_connector, "_transform_to_category_record_group", return_value=mock_rg):
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([
                    {
                        "sys_id": "cat2",
                        "label": "Subcategory",
                        "parent_table": "kb_category",
                        "parent_id": "cat1",
                        "sys_updated_on": "2024-01-01",
                    },
                ])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_categories()
            assert mock_rg.parent_external_group_id == "cat1"


# ===========================================================================
# Deep sync: _fetch_all_groups pagination
# ===========================================================================

    @pytest.mark.asyncio
    async def test_sync_categories_pagination(self, servicenow_connector, mock_data_entities_processor):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        page1 = [
            _category_api_row(sys_id=f"cat{i}", label=f"Cat {i}", value=f"cat-{i}", sys_updated_on="2024-01-01")
            for i in range(100)
        ]
        page2 = [_category_api_row(sys_id="cat100", label="Cat 100", value="cat-100", sys_updated_on="2024-01-02")]

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response(page1),
            _table_api_response(page2),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_categories()
        assert mock_ds.get_now_table_tableName.call_count == 2
        assert mock_data_entities_processor.on_new_record_groups.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_categories_api_error_handling(self, servicenow_connector):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "fail", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._sync_categories()
        servicenow_connector.category_sync_point.update_sync_point.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_categories_skips_invalid_transform(self, servicenow_connector, mock_data_entities_processor):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            _category_api_row(sys_id="cat1", label="", sys_updated_on="2024-01-01"),
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_categories()
        mock_data_entities_processor.on_new_record_groups.assert_not_called()


# ===========================================================================
# _process_record_updates_batch
# ===========================================================================

    @pytest.mark.asyncio
    async def test_sync_categories_delta_sync(self, servicenow_connector):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(
            return_value={"last_sync_time": "2024-01-01 00:00:00"}
        )
        servicenow_connector.category_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_categories()
        call_kwargs = mock_ds.get_now_table_tableName.call_args.kwargs
        assert "2024-01-01" in call_kwargs["sysparm_query"]

    @pytest.mark.asyncio
    async def test_sync_categories_exception_propagates(self, servicenow_connector):
        servicenow_connector.category_sync_point = AsyncMock()
        servicenow_connector.category_sync_point.read_sync_point = AsyncMock(
            side_effect=RuntimeError("category sync fail")
        )
        with pytest.raises(RuntimeError, match="category sync fail"):
            await servicenow_connector._sync_categories()


class TestFetchAllGroupsDeep:
    async def test_paginates_groups(self, servicenow_connector):
        page1 = [{"sys_id": f"g{i}", "name": f"Group {i}"} for i in range(100)]
        page2 = [{"sys_id": "g100", "name": "Group 100"}]

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(side_effect=[
                _table_api_response(page1),
                _table_api_response(page2),
            ])
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_groups()
            assert len(result) == 101

    async def test_handles_exception(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock,
                          side_effect=Exception("API down")):
            with pytest.raises(Exception, match="API down"):
                await servicenow_connector._fetch_all_groups()


# ===========================================================================
# Deep sync: _fetch_all_memberships pagination
# ===========================================================================

class TestFetchAllMembershipsDeep:
    @pytest.mark.asyncio
    async def test_a_failed_page_does_not_yield_a_partial_membership_list(self, servicenow_connector):
        """A partial list makes the caller delete edges it cannot rebuild."""
        sync_point = AsyncMock()
        sync_point.read_sync_point = AsyncMock(return_value=None)
        sync_point.update_sync_point = AsyncMock()
        servicenow_connector.group_sync_point = sync_point

        full_page = _table_api_response([
            {"sys_id": f"m{i}", "user": f"u{i}", "group": "g1", "sys_updated_on": "2026-08-01 10:00:00"}
            for i in range(100)
        ])
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=[full_page, ServiceNowAPIError(500, "page two failed", None)]
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._fetch_all_memberships()

    async def test_paginates_memberships(self, servicenow_connector):
        page1 = [{"sys_id": f"m{i}", "user": f"u{i}", "group": "g1", "sys_updated_on": "2024-01-01"} for i in range(100)]
        page2 = [{"sys_id": "m100", "user": "u100", "group": "g1", "sys_updated_on": "2024-01-02"}]

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(side_effect=[
                _table_api_response(page1),
                _table_api_response(page2),
            ])
            mock_ds.return_value = mock_datasource
            result = await servicenow_connector._fetch_all_memberships()
            assert len(result) == 101

    async def test_handles_exception(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock,
                          side_effect=Exception("API down")):
            with pytest.raises(Exception, match="API down"):
                await servicenow_connector._fetch_all_memberships()


# ===========================================================================
# Deep sync: _get_admin_users with dict ref
# ===========================================================================

    @pytest.mark.asyncio
    async def test_fetch_all_memberships_api_error_propagates(self, servicenow_connector):
        """A partial membership list would silently remove users from their groups."""
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "fail", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._fetch_all_memberships()


class TestGetAdminUsersDeep:
    async def test_handles_string_user_ref(self, servicenow_connector):
        mock_app_user = MagicMock(spec=AppUser)
        mock_app_user.email = "admin2@example.com"

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([{"user": "string-sys-id"}])
            )
            mock_ds.return_value = mock_datasource

            servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(return_value=mock_app_user)

            result = await servicenow_connector._get_admin_users()
            assert len(result) == 1

    async def test_handles_empty_user_ref(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([{"user": ""}])
            )
            mock_ds.return_value = mock_datasource

            tx = _make_mock_tx_store()

            @asynccontextmanager
            async def _tx():
                yield tx

            servicenow_connector.data_store_provider = MagicMock()
            servicenow_connector.data_store_provider.transaction = _tx

            result = await servicenow_connector._get_admin_users()
            assert result == []

    @pytest.mark.asyncio
    async def test_match_exception_continues(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            return_value=_table_api_response([{"user": "admin-1"}])
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(
            side_effect=RuntimeError("lookup fail")
        )

        with patch.object(servicenow_connector.logger, "warning") as mock_warn:
            result = await servicenow_connector._get_admin_users()
        assert result == []
        mock_warn.assert_called()


# ===========================================================================
# Deep sync: _sync_users error handling
# ===========================================================================

class TestSyncUsersErrors:
    async def test_api_error_breaks_loop(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.user_sync_point.update_sync_point = AsyncMock()

        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                side_effect=ServiceNowAPIError(401, "Unauthorized", None)
            )
            mock_ds.return_value = mock_datasource
            with pytest.raises(ServiceNowAPIError):
                await servicenow_connector._sync_users()
            servicenow_connector.data_entities_processor.on_new_app_users.assert_not_called()
            # A page that failed must not leave a checkpoint claiming it was read.
            servicenow_connector.user_sync_point.update_sync_point.assert_not_awaited()

    async def test_exception_propagated(self, servicenow_connector):
        servicenow_connector.user_sync_point = AsyncMock()
        servicenow_connector.user_sync_point.read_sync_point = AsyncMock(
            side_effect=Exception("sync point error")
        )
        with pytest.raises(Exception, match="sync point error"):
            await servicenow_connector._sync_users()


# ===========================================================================
# Deep sync: knowledge bases delta sync
# ===========================================================================

class TestSyncKnowledgeBasesDeep:
    async def test_reads_every_knowledge_base(self, servicenow_connector):
        """A criterion change does not touch the base row, so a delta would miss it."""
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                return_value=_table_api_response([])
            )
            mock_ds.return_value = mock_datasource
            await servicenow_connector._sync_knowledge_bases([])

    async def test_api_error_kbs(self, servicenow_connector):
        with patch.object(servicenow_connector, "_get_fresh_datasource", new_callable=AsyncMock) as mock_ds:
            mock_datasource = AsyncMock()
            mock_datasource.get_now_table_tableName = AsyncMock(
                side_effect=ServiceNowAPIError(500, "Server error", None)
            )
            mock_ds.return_value = mock_datasource
            with pytest.raises(ServiceNowAPIError):
                await servicenow_connector._sync_knowledge_bases([])

    async def test_exception_in_kb_sync_propagated(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(
            side_effect=Exception("kb error")
        )
        with pytest.raises(Exception, match="kb error"):
            await servicenow_connector._sync_knowledge_bases([])

    @pytest.mark.asyncio
    async def test_sync_knowledge_bases_pagination(self, servicenow_connector, mock_data_entities_processor):
        page1 = [
            _kb_api_row(sys_id=f"kb{i}", title=f"KB {i}", sys_updated_on="2024-01-01")
            for i in range(100)
        ]
        page2 = [_kb_api_row(sys_id="kb100", title="KB 100", sys_updated_on="2024-01-02")]

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response(page1),
            _table_api_response(page2),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with patch.object(servicenow_connector, "_fetch_kb_permissions_from_criteria",
                          new_callable=AsyncMock, return_value={"read": [], "write": []}), \
             patch.object(servicenow_connector, "_process_criteria_permissions",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock, return_value=[]):
            await servicenow_connector._sync_knowledge_bases([])

        assert mock_ds.get_now_table_tableName.call_count == 2
        mock_data_entities_processor.on_new_record_groups.assert_called()

    @pytest.mark.asyncio
    async def test_sync_knowledge_bases_owner_permission(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            _kb_api_row(sys_id="kb1", title="KB 1", owner="owner1", sys_updated_on="2024-01-01"),
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        owner_perm = Permission(email="owner@test.com", type=PermissionType.OWNER, entity_type=EntityType.USER)

        with patch.object(servicenow_connector, "_fetch_kb_permissions_from_criteria",
                          new_callable=AsyncMock, return_value={"read": [], "write": []}), \
             patch.object(servicenow_connector, "_process_criteria_permissions",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock, return_value=[owner_perm]):
            await servicenow_connector._sync_knowledge_bases([])

    @pytest.mark.asyncio
    async def test_sync_knowledge_bases_transform_returns_none_skipped(self, servicenow_connector, mock_data_entities_processor):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            {"sys_id": "kb1", "title": "", "sys_updated_on": "2024-01-01"},
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_knowledge_bases([])
        mock_data_entities_processor.on_new_record_groups.assert_not_called()


# ===========================================================================
# _sync_categories (additional)
# ===========================================================================


class TestConvertPermissionsToObjects:
    @pytest.mark.asyncio
    async def test_convert_permissions_user_type(self, servicenow_connector):
        mock_user = MagicMock(spec=AppUser)
        mock_user.email = "user@test.com"
        servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(return_value=mock_user)

        perms = await servicenow_connector._convert_permissions_to_objects(
            [RawPermission(entity_type="USER", source_sys_id="u1", role="READER")],
        )
        assert len(perms) == 1
        assert perms[0].email == "user@test.com"
        assert perms[0].type == PermissionType.READ

    @pytest.mark.asyncio
    async def test_convert_permissions_user_not_found(self, servicenow_connector):
        perms = await servicenow_connector._convert_permissions_to_objects(
            [RawPermission(entity_type="USER", source_sys_id="missing", role="READER")],
        )
        assert perms == []

    @pytest.mark.asyncio
    async def test_convert_permissions_group_type(self, servicenow_connector):
        perms = await servicenow_connector._convert_permissions_to_objects(
            [RawPermission(entity_type="GROUP", source_sys_id="g1", role="WRITER")],
        )
        assert len(perms) == 1
        assert perms[0].external_id == "g1"
        assert perms[0].type == PermissionType.WRITE

    @pytest.mark.asyncio
    async def test_convert_permissions_unknown_entity_type(self, servicenow_connector):
        bad_perm = MagicMock()
        bad_perm.entity_type = "ROLE"
        bad_perm.source_sys_id = "r1"
        bad_perm.role = "READER"

        perms = await servicenow_connector._convert_permissions_to_objects([bad_perm])
        assert perms == []

    @pytest.mark.asyncio
    async def test_convert_permissions_exception_handling(self, servicenow_connector):
        servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(side_effect=RuntimeError("db error"))

        perms = await servicenow_connector._convert_permissions_to_objects(
            [RawPermission(entity_type="USER", source_sys_id="u1", role="READER")],
        )
        assert perms == []

    @pytest.mark.asyncio
    async def test_convert_permissions_mixed_types(self, servicenow_connector):
        mock_user = MagicMock(spec=AppUser)
        mock_user.email = "user@test.com"
        servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(return_value=mock_user)

        perms = await servicenow_connector._convert_permissions_to_objects(
            [
                RawPermission(entity_type="USER", source_sys_id="u1", role="READER"),
                RawPermission(entity_type="GROUP", source_sys_id="g1", role="WRITER"),
            ],
        )
        assert len(perms) == 2

    @pytest.mark.asyncio
    async def test_convert_permissions_empty_list(self, servicenow_connector):
        perms = await servicenow_connector._convert_permissions_to_objects([])
        assert perms == []


class TestExtractPermissionsFromUserCriteria:
    @pytest.mark.asyncio
    async def test_extract_permissions_user_field(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", user="u1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert len(perms) == 1
        assert perms[0].entity_type == "USER"
        assert perms[0].source_sys_id == "u1"

    @pytest.mark.asyncio
    async def test_extract_permissions_group_field(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", group="g1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.WRITE
        )
        assert perms[0].entity_type == "GROUP"
        assert perms[0].role == "WRITER"

    @pytest.mark.asyncio
    async def test_extract_permissions_role_field(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", role="r1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert perms[0].entity_type == "GROUP"
        assert perms[0].source_sys_id == "r1"

    @pytest.mark.asyncio
    async def test_extract_permissions_department_field(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", department="dept1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert perms[0].source_sys_id == "dept1"

    @pytest.mark.asyncio
    async def test_extract_permissions_location_field(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", location="loc1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert perms[0].source_sys_id == "loc1"

    @pytest.mark.asyncio
    async def test_extract_permissions_company_field(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", company="comp1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert perms[0].source_sys_id == "comp1"

    @pytest.mark.asyncio
    async def test_extract_permissions_multiple_comma_separated(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", user="u1,u2, u3")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert len(perms) == 3
        assert {p.source_sys_id for p in perms} == {"u1", "u2", "u3"}

    @pytest.mark.asyncio
    async def test_extract_permissions_empty_fields(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert perms == []

    @pytest.mark.asyncio
    async def test_extract_permissions_mixed_fields(self, servicenow_connector):
        criteria = UserCriteria(sys_id="c1", user="u1", group="g1", company="c1")
        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            criteria, PermissionType.READ
        )
        assert len(perms) == 3

    @pytest.mark.asyncio
    async def test_extract_permissions_exception_handling(self, servicenow_connector):
        class BadCriteria:
            sys_id = "c1"

            @property
            def user(self):
                raise RuntimeError("bad")

        perms = await servicenow_connector._extract_permissions_from_user_criteria_details(
            BadCriteria(), PermissionType.READ
        )
        assert perms == []


# ===========================================================================
# _transform_to_kb_record_group / _transform_to_category_record_group (additional)
# ===========================================================================


class TestCreateConnector:
    @pytest.mark.asyncio
    async def test_create_connector_factory(self):
        with patch("app.connectors.sources.servicenow.servicenow.connector.ServicenowApp"):
            processor = MagicMock()
            processor.org_id = "org-1"

            result = await ServiceNowConnector.create_connector(
                logger=logging.getLogger("test"),
                data_store_provider=MagicMock(),
                config_service=AsyncMock(),
                connector_id="factory-1",
                scope="team",
                created_by="user-1",
                data_entities_processor=processor,
            )
            assert isinstance(result, ServiceNowConnector)

    @pytest.mark.asyncio
    async def test_reindex_records_logs_warning(self, servicenow_connector):
        with patch.object(servicenow_connector.logger, "warning") as mock_warn:
            await servicenow_connector.reindex_records([])
            mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_exception_handling(self, servicenow_connector):
        servicenow_connector.servicenow_client = MagicMock()
        servicenow_connector.servicenow_datasource = MagicMock()
        with patch.object(servicenow_connector.logger, "info", side_effect=[None, RuntimeError("cleanup fail")]):
            await servicenow_connector.cleanup()
        assert servicenow_connector.servicenow_client is None


class TestFetchAttachmentsForArticle:
    @pytest.mark.asyncio
    async def test_fetch_attachments_success(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            {
                "sys_id": "att1",
                "file_name": "file.pdf",
                "content_type": "application/pdf",
                "size_bytes": "100",
                "table_sys_id": "art1",
                "sys_created_on": "2023-01-01 00:00:00",
                "sys_updated_on": "2023-06-01 00:00:00",
            },
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_attachments_for_article("art1")
        assert len(result) == 1
        assert result[0].file_name == "file.pdf"

    @pytest.mark.asyncio
    async def test_fetch_attachments_no_results(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_attachments_for_article("art1")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_attachments_api_error(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(404, "Not found", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_attachments_for_article("art1")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_attachments_general_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("network"))

        result = await servicenow_connector._fetch_attachments_for_article("art1")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_attachments_multiple_attachments(self, servicenow_connector):
        records = [
            {
                "sys_id": f"att{i}",
                "file_name": f"file{i}.pdf",
                "content_type": "application/pdf",
                "size_bytes": "100",
                "table_sys_id": "art1",
                "sys_created_on": "2023-01-01 00:00:00",
                "sys_updated_on": "2023-06-01 00:00:00",
            }
            for i in range(12)
        ]
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response(records))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_attachments_for_article("art1")
        assert len(result) == 12


# ===========================================================================
# _convert_permissions_to_objects and criteria helpers
# ===========================================================================


class TestFetchKbPermissionsFromCriteria:
    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_read_criteria(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response([{"user_criteria": "crit-read-1"}]),
            _table_api_response([]),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert "crit-read-1" in result["read"]

    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_write_criteria(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response([]),
            _table_api_response([{"user_criteria": "crit-write-1"}]),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert "crit-write-1" in result["write"]

    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_both_read_and_write(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response([{"user_criteria": "crit-r"}]),
            _table_api_response([{"user_criteria": "crit-w"}]),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert result["read"] == ["crit-r"]
        assert result["write"] == ["crit-w"]

    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_read_api_error(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            ServiceNowAPIError(500, "read fail", None),
            _table_api_response([{"user_criteria": "crit-w"}]),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert result["read"] == []
        assert result["write"] == ["crit-w"]

    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_write_api_error(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response([{"user_criteria": "crit-r"}]),
            ServiceNowAPIError(500, "write fail", None),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert result["read"] == ["crit-r"]
        assert result["write"] == []

    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_general_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("boom"))

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert result == {"read": [], "write": []}

    @pytest.mark.asyncio
    async def test_fetch_kb_permissions_empty_results(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_kb_permissions_from_criteria("kb1")
        assert result == {"read": [], "write": []}


class TestFlattenAndRoles:
    @pytest.mark.asyncio
    async def test_flatten_and_create_user_groups_deep_hierarchy(self, servicenow_connector):
        groups = [
            SysUserGroup(sys_id="g1", name="Root"),
            SysUserGroup(sys_id="g2", name="Child", parent="g1"),
            SysUserGroup(sys_id="g3", name="Grandchild", parent="g2"),
        ]
        memberships = [
            SysUserGroupMembership(sys_id="m1", user="u1", group="g3"),
        ]
        mock_user = MagicMock(spec=AppUser)
        mock_user.source_user_id = "u1"

        servicenow_connector.data_entities_processor.get_all_app_users = AsyncMock(return_value=[mock_user])

        with patch.object(servicenow_connector, "_transform_to_user_group") as mock_transform:
            mock_group = MagicMock(spec=AppUserGroup)
            mock_transform.return_value = mock_group
            result = await servicenow_connector._flatten_and_create_user_groups(groups, memberships)

        assert len(result) == 3
        assert any(mock_user in users for _, users in result)

    @pytest.mark.asyncio
    async def test_flatten_and_create_user_groups_circular_reference(self, servicenow_connector):
        groups = [
            SysUserGroup(sys_id="g1", name="A", parent="g2"),
            SysUserGroup(sys_id="g2", name="B", parent="g1"),
        ]
        memberships = []

        servicenow_connector.data_entities_processor.get_all_app_users = AsyncMock(return_value=[])

        with patch.object(servicenow_connector, "_transform_to_user_group") as mock_transform:
            mock_transform.return_value = MagicMock(spec=AppUserGroup)
            result = await servicenow_connector._flatten_and_create_user_groups(groups, memberships)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_all_role_assignments_pagination(self, servicenow_connector):
        page1 = [
            {"sys_id": f"ra{i}", "user": f"u{i}", "role": "r1", "sys_updated_on": "2024-01-01"}
            for i in range(100)
        ]
        page2 = [{"sys_id": "ra100", "user": "u100", "role": "r1", "sys_updated_on": "2024-01-02"}]

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response(page1),
            _table_api_response(page2),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_all_role_assignments()
        assert len(result) == 101

    @pytest.mark.asyncio
    async def test_fetch_all_role_assignments_api_error_propagates(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "fail", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._fetch_all_role_assignments()

    @pytest.mark.asyncio
    async def test_fetch_role_hierarchy_empty_results(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._fetch_role_hierarchy()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_role_hierarchy_api_error_propagates(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "fail", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._fetch_role_hierarchy()

    @pytest.mark.asyncio
    async def test_fetch_all_roles_api_error_propagates(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "fail", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._fetch_all_roles()


# ===========================================================================
# _sync_knowledge_bases (additional)
# ===========================================================================


class TestInitRefreshToken:
    @pytest.mark.asyncio
    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowRESTClientViaOAuthAuthorizationCode")
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowDataSource")
    async def test_init_with_refresh_token(self, mock_ds_cls, mock_client_cls, mock_fetch, servicenow_connector):
        mock_fetch.return_value = {
            "config": {
                "clientId": "cid",
                "clientSecret": "secret",
                "instanceUrl": "https://sn.example.com",
            },
            "redirectUri": "http://localhost/callback",
        }
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_ds_cls.return_value = MagicMock()

        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "auth": {"oauthConfigId": "oauth-1"},
            "credentials": {
                "access_token": "token",
                "refresh_token": "refresh-tok",
            },
        })
        servicenow_connector.test_connection_and_access = AsyncMock(return_value=True)

        assert await servicenow_connector.init() is True
        assert mock_client.refresh_token == "refresh-tok"

    @pytest.mark.asyncio
    @patch("app.utils.oauth_config.fetch_oauth_config_by_id", new_callable=AsyncMock)
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowRESTClientViaOAuthAuthorizationCode")
    @patch("app.connectors.sources.servicenow.servicenow.connector.ServiceNowDataSource")
    async def test_init_without_refresh_token(self, mock_ds_cls, mock_client_cls, mock_fetch, servicenow_connector):
        mock_fetch.return_value = {
            "config": {
                "clientId": "cid",
                "clientSecret": "secret",
                "instanceUrl": "https://sn.example.com",
            },
            "redirectUri": "http://localhost/callback",
        }
        mock_client = MagicMock()
        mock_client.refresh_token = None
        mock_client_cls.return_value = mock_client
        mock_ds_cls.return_value = MagicMock()

        servicenow_connector.config_service.get_config = AsyncMock(return_value={
            "auth": {"oauthConfigId": "oauth-1"},
            "credentials": {"access_token": "token"},
        })
        servicenow_connector.test_connection_and_access = AsyncMock(return_value=True)

        assert await servicenow_connector.init() is True
        assert mock_client.refresh_token is None

    @pytest.mark.asyncio
    async def test_handle_webhook_exception_returns_false(self, servicenow_connector):
        with patch.object(servicenow_connector.logger, "info", side_effect=RuntimeError("log fail")):
            result = await servicenow_connector.handle_webhook_notification("org-1", {"event": "test"})
        assert result is False


# ===========================================================================
# stream_record generators
# ===========================================================================


class TestProcessCriteriaPermissions:
    @pytest.mark.asyncio
    async def test_process_criteria_permissions_empty_criteria_ids(self, servicenow_connector):
        result = await servicenow_connector._process_criteria_permissions([], PermissionType.READ)
        assert result == []

    @pytest.mark.asyncio
    async def test_process_criteria_permissions_batch_fetch_success(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            {"sys_id": "crit1", "user": "u1", "group": "", "role": "",
             "department": "", "location": "", "company": "", "cost_center": ""},
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with patch.object(servicenow_connector, "_extract_permissions_from_user_criteria_details",
                          new_callable=AsyncMock, return_value=[
                              RawPermission(entity_type="USER", source_sys_id="u1", role="READER"),
                          ]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock, return_value=[
                              Permission(email="u@test.com", type=PermissionType.READ, entity_type=EntityType.USER),
                          ]):
            result = await servicenow_connector._process_criteria_permissions(["crit1"], PermissionType.READ)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_process_criteria_permissions_api_error(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "fail", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock, return_value=[]):
            result = await servicenow_connector._process_criteria_permissions(["crit1"], PermissionType.READ)

        assert result == []

    @pytest.mark.asyncio
    async def test_process_criteria_permissions_extract_and_convert(self, servicenow_connector):
        mock_user = MagicMock(spec=AppUser)
        mock_user.email = "extracted@test.com"
        servicenow_connector.data_entities_processor.get_user_by_source_id = AsyncMock(return_value=mock_user)

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            {"sys_id": "crit1", "user": "u1"},
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        result = await servicenow_connector._process_criteria_permissions(["crit1"], PermissionType.READ)
        assert len(result) == 1
        assert result[0].email == "extracted@test.com"

    @pytest.mark.asyncio
    async def test_process_criteria_permissions_general_exception(self, servicenow_connector):
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("boom"))

        result = await servicenow_connector._process_criteria_permissions(["crit1"], PermissionType.READ)
        assert result == []


class TestProcessRecordUpdatesBatch:
    @pytest.mark.asyncio
    async def test_process_record_updates_batch_empty_list(self, servicenow_connector, mock_data_entities_processor):
        await servicenow_connector._process_record_updates_batch([])
        mock_data_entities_processor.on_new_records.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_record_updates_batch_success(self, servicenow_connector, mock_data_entities_processor):
        record = MagicMock(spec=WebpageRecord)
        perm = Permission(email="u@test.com", type=PermissionType.READ, entity_type=EntityType.USER)
        update = _record_update(
            record=record,
            new_permissions=[perm],
            external_record_id="art1",
        )
        await servicenow_connector._process_record_updates_batch([update])
        mock_data_entities_processor.on_new_records.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_record_updates_batch_skips_missing_permissions(self, servicenow_connector, mock_data_entities_processor):
        record = MagicMock(spec=WebpageRecord)
        update = _record_update(
            record=record,
            permissions_changed=False,
            new_permissions=[],
            external_record_id="art1",
        )
        await servicenow_connector._process_record_updates_batch([update])
        mock_data_entities_processor.on_new_records.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_record_updates_batch_reports_dropped_records(self, servicenow_connector, mock_data_entities_processor):
        """A dropped record is never written, so the caller must be able to count it."""
        granted = _record_update(
            record=MagicMock(spec=WebpageRecord),
            new_permissions=[Permission(email="u@test.com", type=PermissionType.READ, entity_type=EntityType.USER)],
            external_record_id="art1",
        )
        dropped = _record_update(
            record=MagicMock(spec=WebpageRecord),
            new_permissions=[],
            external_record_id="art2",
        )
        servicenow_connector.logger = MagicMock()

        dropped_count = await servicenow_connector._process_record_updates_batch([granted, dropped])

        assert dropped_count == 1
        written = mock_data_entities_processor.on_new_records.call_args.args[0]
        assert len(written) == 1
        assert servicenow_connector.logger.warning.call_args.args[2] == ["art2"]


# ===========================================================================
# create_connector and cleanup
# ===========================================================================

    @pytest.mark.asyncio
    async def test_process_record_updates_batch_exception(self, servicenow_connector):
        record = MagicMock(spec=WebpageRecord)
        perm = Permission(email="u@test.com", type=PermissionType.READ, entity_type=EntityType.USER)
        update = _record_update(record=record, new_permissions=[perm], external_record_id="art1")
        servicenow_connector.data_entities_processor.on_new_records = AsyncMock(
            side_effect=RuntimeError("batch fail")
        )
        with pytest.raises(RuntimeError, match="batch fail"):
            await servicenow_connector._process_record_updates_batch([update])


class TestProcessSingleArticle:
    @pytest.mark.asyncio
    async def test_process_single_article_with_attachments(self, servicenow_connector):
        article = KBKnowledge(
            sys_id="art1",
            short_description="Article with attachment",
            kb_category="cat1",
            author="author1",
        )
        att = AttachmentMetadata(
            sys_id="att1",
            file_name="doc.pdf",
            content_type="application/pdf",
            size_bytes="1024",
            table_sys_id="art1",
            table_name="kb_knowledge",
            sys_created_on="2023-01-01 00:00:00",
            sys_updated_on="2023-06-01 00:00:00",
        )

        with patch.object(servicenow_connector, "_fetch_attachments_for_article",
                          new_callable=AsyncMock, return_value=[att]), \
             patch.object(servicenow_connector, "_process_criteria_permissions",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock, return_value=[]):
            updates = await servicenow_connector._process_single_article(article)

        assert len(updates) == 2
        assert updates[0].record.record_type == RecordType.WEBPAGE
        assert updates[1].record.record_type == RecordType.FILE

    @pytest.mark.asyncio
    async def test_process_single_article_with_permissions(self, servicenow_connector):
        article = KBKnowledge(
            sys_id="art2",
            short_description="Article with criteria",
            kb_category="cat1",
            can_read_user_criteria="crit1,crit2",
        )
        mock_perm = Permission(email="user@test.com", type=PermissionType.READ, entity_type=EntityType.USER)

        with patch.object(servicenow_connector, "_fetch_attachments_for_article",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_process_criteria_permissions",
                          new_callable=AsyncMock, return_value=[mock_perm]):
            updates = await servicenow_connector._process_single_article(article)

        assert len(updates) == 1
        assert mock_perm in updates[0].new_permissions

    @pytest.mark.asyncio
    async def test_process_single_article_with_author_owner(self, servicenow_connector):
        article = KBKnowledge(
            sys_id="art3",
            short_description="Article with author",
            kb_category="cat1",
            author="author-sys-id",
        )
        owner_perm = Permission(email="author@test.com", type=PermissionType.OWNER, entity_type=EntityType.USER)

        with patch.object(servicenow_connector, "_fetch_attachments_for_article",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_process_criteria_permissions",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock, return_value=[owner_perm]):
            updates = await servicenow_connector._process_single_article(article)

        assert owner_perm in updates[0].new_permissions

    @pytest.mark.asyncio
    async def test_process_single_article_without_author(self, servicenow_connector):
        article = KBKnowledge(
            sys_id="art4",
            short_description="No author",
            kb_category="cat1",
            author=None,
        )

        with patch.object(servicenow_connector, "_fetch_attachments_for_article",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_process_criteria_permissions",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(servicenow_connector, "_convert_permissions_to_objects",
                          new_callable=AsyncMock) as mock_convert:
            updates = await servicenow_connector._process_single_article(article)

        mock_convert.assert_not_called()
        assert len(updates) == 1

    @pytest.mark.asyncio
    async def test_process_single_article_transform_fails(self, servicenow_connector):
        article = KBKnowledge(
            sys_id="art5",
            short_description="",
            kb_category="cat1",
        )
        updates = await servicenow_connector._process_single_article(article)
        assert updates == []

    @pytest.mark.asyncio
    async def test_process_single_article_exception_handling(self, servicenow_connector):
        article = TableAPIRecord(
            sys_id="art6",
            short_description="Will fail",
            kb_category="cat1",
            can_read_user_criteria="",
            author=None,
        )

        with patch.object(servicenow_connector, "_fetch_attachments_for_article",
                          new_callable=AsyncMock, side_effect=RuntimeError("fetch fail")):
            updates = await servicenow_connector._process_single_article(article)

        assert updates == []


# ===========================================================================
# _fetch_attachments_for_article
# ===========================================================================

    @pytest.mark.asyncio
    async def test_process_single_article_exception_returns_empty(self, servicenow_connector):
        article = TableAPIRecord(sys_id="art1", short_description="Title", kb_category="cat1")
        with patch.object(
            servicenow_connector, "_transform_to_article_webpage_record",
            side_effect=RuntimeError("transform fail"),
        ):
            result = await servicenow_connector._process_single_article(article)
        assert result == []


class TestStreamRecordGenerators:
    @pytest.mark.asyncio
    async def test_stream_article_generator_yields_content(self, servicenow_connector):
        servicenow_connector._fetch_article_content = AsyncMock(return_value="<p>Hello</p>")
        record = MagicMock()
        record.record_type = RecordType.WEBPAGE
        record.record_name = "Article"
        record.external_record_id = "art-1"

        response = await servicenow_connector.stream_record(record)
        body = b"".join([chunk async for chunk in response.body_iterator])
        assert b"Hello" in body

    @pytest.mark.asyncio
    async def test_stream_attachment_generator_yields_bytes(self, servicenow_connector):
        servicenow_connector._fetch_attachment_content = AsyncMock(return_value=b"file-bytes")
        record = MagicMock()
        record.record_type = RecordType.FILE
        record.record_name = "file.pdf"
        record.external_record_id = "att-1"
        record.mime_type = "application/pdf"
        record.id = "rec-1"

        response = await servicenow_connector.stream_record(record)
        body = b"".join([chunk async for chunk in response.body_iterator])
        assert b"file-bytes" in body


# ===========================================================================
# role assignments and hierarchy fetch
# ===========================================================================


class TestSyncArticles:
    @pytest.mark.asyncio
    async def test_sync_articles_full_sync_with_pagination(self, servicenow_connector, mock_data_entities_processor):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.article_sync_point.update_sync_point = AsyncMock()

        page1 = [
            {"sys_id": f"art{i}", "short_description": f"Article {i}",
             "kb_category": "cat1", "sys_updated_on": "2024-01-01"}
            for i in range(100)
        ]
        page2 = [
            {"sys_id": "art100", "short_description": "Article 100",
             "kb_category": "cat1", "sys_updated_on": "2024-01-02"},
        ]

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response(page1),
            _table_api_response(page2),
            _table_api_response([]),   # the unpublished pass, with its own projection
            _table_api_response([]),   # the sys_audit_delete pass
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with patch.object(servicenow_connector, "_process_single_article", new_callable=AsyncMock,
                          return_value=[_record_update()]):
            await servicenow_connector._sync_articles()

        # Two indexing pages, then the two removal passes.
        assert mock_ds.get_now_table_tableName.call_count == 4
        servicenow_connector.article_sync_point.update_sync_point.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_articles_delta_sync_with_last_sync_time(self, servicenow_connector):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(
            return_value={"last_sync_time": "2024-01-01 00:00:00"}
        )
        servicenow_connector.article_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_articles()
        call_kwargs = mock_ds.get_now_table_tableName.call_args.kwargs
        assert "2024-01-01" in call_kwargs["sysparm_query"]

    @pytest.mark.asyncio
    async def test_sync_articles_api_error_breaks_loop(self, servicenow_connector):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.article_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            side_effect=ServiceNowAPIError(500, "Server error", None)
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with pytest.raises(ServiceNowAPIError):
            await servicenow_connector._sync_articles()
        mock_data_entities_processor = servicenow_connector.data_entities_processor
        mock_data_entities_processor.on_new_records.assert_not_called()
        servicenow_connector.article_sync_point.update_sync_point.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_articles_updates_sync_checkpoint(self, servicenow_connector):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.article_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            {"sys_id": "art1", "short_description": "A1", "kb_category": "c1",
             "sys_updated_on": "2024-06-15 12:00:00"},
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        with patch.object(servicenow_connector, "_process_single_article", new_callable=AsyncMock, return_value=[]):
            await servicenow_connector._sync_articles()

        servicenow_connector.article_sync_point.update_sync_point.assert_called_once()
        checkpoint = servicenow_connector.article_sync_point.update_sync_point.call_args[0][1]
        assert checkpoint["last_sync_time"] == "2024-06-15 12:00:00"

    @pytest.mark.asyncio
    async def test_sync_articles_empty_result(self, servicenow_connector):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.article_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        await servicenow_connector._sync_articles()
        servicenow_connector.article_sync_point.update_sync_point.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_articles_counts_attachments_correctly(self, servicenow_connector):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(return_value=None)
        servicenow_connector.article_sync_point.update_sync_point = AsyncMock()

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(return_value=_table_api_response([
            {"sys_id": "art1", "short_description": "A1", "kb_category": "c1",
             "sys_updated_on": "2024-01-01"},
        ]))
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        article_update = _record_update(record=MagicMock(record_type=RecordType.WEBPAGE))
        file_update = _record_update(record=MagicMock(record_type=RecordType.FILE))

        with patch.object(servicenow_connector, "_process_single_article", new_callable=AsyncMock,
                          return_value=[article_update, file_update]) as mock_process_single, \
             patch.object(servicenow_connector, "_process_record_updates_batch", new_callable=AsyncMock) as mock_batch:
            await servicenow_connector._sync_articles()
            
            mock_process_single.assert_called_once()
            mock_batch.assert_called_once()
            batch_call_args = mock_batch.call_args[0][0]
            assert len(batch_call_args) == 2
            assert batch_call_args[0].record.record_type == RecordType.WEBPAGE
            assert batch_call_args[1].record.record_type == RecordType.FILE

    @pytest.mark.asyncio
    async def test_sync_articles_exception_propagates(self, servicenow_connector):
        servicenow_connector.article_sync_point = AsyncMock()
        servicenow_connector.article_sync_point.read_sync_point = AsyncMock(
            side_effect=RuntimeError("article sync fail")
        )
        with pytest.raises(RuntimeError, match="article sync fail"):
            await servicenow_connector._sync_articles()


# ===========================================================================
# _resolve_kb_portal_suffix and the article web URL
# ===========================================================================

class TestResolveKbPortalSuffix:
    @pytest.mark.asyncio
    async def test_uses_the_instance_property(self, servicenow_connector):
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(
            return_value=_table_api_response([{"value": "kb"}])
        )
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        assert await servicenow_connector._resolve_kb_portal_suffix() == "kb"
        assert mock_ds.get_now_table_tableName.await_count == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_the_default_portal(self, servicenow_connector):
        """An empty property means the default portal serves the knowledge pages."""
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=[
            _table_api_response([{"value": ""}]),
            _table_api_response([{"url_suffix": "esc"}]),
        ])
        servicenow_connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)

        assert await servicenow_connector._resolve_kb_portal_suffix() == "esc"

    @pytest.mark.asyncio
    async def test_falls_back_to_the_knowledge_portal_on_error(self, servicenow_connector):
        """A wrong link is better than a failed sync."""
        servicenow_connector._get_fresh_datasource = AsyncMock(side_effect=RuntimeError("no access"))

        assert await servicenow_connector._resolve_kb_portal_suffix() == "kb"


class TestArticleWebUrl:
    def test_article_url_uses_the_resolved_portal(self, servicenow_connector):
        servicenow_connector.instance_url = "https://dev194883.service-now.com"
        servicenow_connector.kb_portal_suffix = "esc"

        record = servicenow_connector._transform_to_article_webpage_record(
            KBKnowledge(sys_id="art1", short_description="Title", kb_category="cat1")
        )

        assert record.weburl == (
            "https://dev194883.service-now.com/esc?id=kb_article_view&sys_kb_id=art1"
        )


# ===========================================================================
# glide.knowman.apply_article_read_criteria
# ===========================================================================

class TestRecordRevision:
    """A stored record is rewritten, and re-queued for indexing, only when
    external_revision_id changes. Leaving it unset froze every record at its
    first index: title, webUrl and content never updated again."""

    def test_article_carries_a_revision_id(self, servicenow_connector):
        record = servicenow_connector._transform_to_article_webpage_record(
            KBKnowledge(
                sys_id="art1",
                short_description="Title",
                kb_category="cat1",
                sys_updated_on="2026-08-26 09:15:00",
            )
        )
        assert record.external_revision_id == "2026-08-26 09:15:00"

    def test_a_changed_article_gets_a_different_revision_id(self, servicenow_connector):
        def build(updated_on):
            return servicenow_connector._transform_to_article_webpage_record(
                KBKnowledge(
                    sys_id="art1",
                    short_description="Title",
                    kb_category="cat1",
                    sys_updated_on=updated_on,
                )
            )

        before = build("2026-08-26 09:15:00")
        after = build("2026-08-26 11:42:00")
        assert before.external_revision_id != after.external_revision_id

    def test_attachment_carries_a_revision_id(self, servicenow_connector):
        record = servicenow_connector._transform_to_attachment_file_record(
            AttachmentMetadata(
                sys_id="att1",
                file_name="doc.pdf",
                content_type="application/pdf",
                size_bytes="1024",
                table_sys_id="art1",
                sys_created_on="2026-08-01 10:00:00",
                sys_updated_on="2026-08-26 11:42:00",
            ),
            parent_record_group_type=RecordGroupType.SERVICENOW_CATEGORY,
            parent_external_record_group_id="cat1",
        )
        assert record.external_revision_id == "2026-08-26 11:42:00"


class TestUnpublishedArticleRemoval:
    """An article that leaves the published set must take its record with it.

    ServiceNow offers no change feed, so a deletion never arrives as an event —
    only as a row saying the article is no longer published. Measured on a
    developer instance: a retired article stayed browsable and searchable, with
    its old text, through every following delta sync."""

    def _article(self, **kw):
        # TableAPIRecord, not KBKnowledge: this is what the sync loop hands over,
        # and it raises AttributeError for a field the response did not carry.
        base = dict(sys_id="art1", short_description="Title", kb_category="cat1",
                    active="true", workflow_state="published")
        base.update(kw)
        return TableAPIRecord(**{k: v for k, v in base.items() if v is not None})

    def test_published_article_stays(self, servicenow_connector):
        assert servicenow_connector._article_left_published_set(self._article()) is False

    @pytest.mark.parametrize("field,value", [
        ("workflow_state", "outdated"),
        ("workflow_state", "retired"),
        ("workflow_state", "draft"),
        ("active", "false"),
    ])
    def test_article_outside_the_published_set(self, servicenow_connector, field, value):
        assert servicenow_connector._article_left_published_set(self._article(**{field: value})) is True

    @pytest.mark.parametrize("row", [
        {"workflow_state": None, "active": None},
        {"workflow_state": "", "active": ""},
    ])
    def test_a_missing_field_never_deletes(self, servicenow_connector, row):
        """Deleting a record cannot be undone by the next sync, so an absent
        field must not stand in for evidence that the article was retired."""
        assert servicenow_connector._article_left_published_set(self._article(**row)) is False

    @pytest.mark.asyncio
    async def test_retired_article_deletes_its_record(self, servicenow_connector,
                                                      mock_data_entities_processor):
        record = MagicMock()
        record.id = "rec-1"
        mock_data_entities_processor.get_record_by_external_id = AsyncMock(return_value=record)
        mock_data_entities_processor.on_records_deleted_cascade = AsyncMock(
            return_value={"deleted_records": ["rec-1"]}
        )

        deleted = await servicenow_connector._remove_article_records("art1")

        assert deleted == 1
        mock_data_entities_processor.on_records_deleted_cascade.assert_awaited_once_with(
            ["rec-1"], servicenow_connector.connector_id
        )

    @pytest.mark.asyncio
    async def test_the_cascade_path_takes_the_attachments_too(self, servicenow_connector,
                                                              mock_data_entities_processor):
        """The cascade deletes the containment subtree, so attachments need no
        separate pass — and unlike on_record_deleted it removes the edges and
        the type document rather than leaving them dangling."""
        record = MagicMock()
        record.id = "rec-1"
        mock_data_entities_processor.get_record_by_external_id = AsyncMock(return_value=record)
        mock_data_entities_processor.on_records_deleted_cascade = AsyncMock(
            return_value={"deleted_records": ["rec-1", "att-1"]}
        )

        deleted = await servicenow_connector._remove_article_records("art1")

        assert deleted == 2
        mock_data_entities_processor.on_record_deleted.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_article_that_was_never_indexed_deletes_nothing(
        self, servicenow_connector, mock_data_entities_processor
    ):
        mock_data_entities_processor.get_record_by_external_id = AsyncMock(return_value=None)

        deleted = await servicenow_connector._remove_article_records("art1")

        assert deleted == 0
        mock_data_entities_processor.on_records_deleted_cascade.assert_not_awaited()


class TestSyncArticlesRemovesUnpublished:
    """The behavioural witness: drive _sync_articles over a retired article and
    require that its record goes. Reproduced live — a retired article stayed
    browsable and searchable through every following delta sync."""

    def _row(self, **kw):
        base = dict(sys_id="art1", short_description="Title", kb_category="cat1",
                    active="true", workflow_state="published",
                    sys_updated_on="2026-08-27 09:38:57")
        base.update(kw)
        return base

    async def _run(self, connector, rows):
        """Answer the removal pass with `rows` and the indexing pass with nothing.

        The two passes hit the same endpoint and are told apart by their
        projection: the removal pass reads only the three fields its decision
        needs, so it never pulls article bodies for retired rows.
        """
        calls = []

        async def _respond(**kwargs):
            calls.append(kwargs)
            fields = kwargs.get("sysparm_fields", "")
            is_removal_pass = fields == "sys_id,workflow_state,active"
            return _table_api_response(rows if is_removal_pass else [])

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=_respond)
        connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)
        connector.article_sync_point.read_sync_point = AsyncMock(
            return_value={"last_sync_time": "2026-08-27 09:30:09"}
        )
        connector.article_sync_point.update_sync_point = AsyncMock()
        await connector._sync_articles()
        return calls

    @pytest.mark.asyncio
    async def test_retired_article_loses_its_record(self, servicenow_connector,
                                                    mock_data_entities_processor):
        record = MagicMock()
        record.id = "rec-1"
        mock_data_entities_processor.get_record_by_external_id = AsyncMock(return_value=record)

        await self._run(servicenow_connector, [self._row(workflow_state="outdated")])

        mock_data_entities_processor.on_records_deleted_cascade.assert_awaited_once_with(
            ["rec-1"], servicenow_connector.connector_id
        )

    @pytest.mark.asyncio
    async def test_published_article_keeps_its_record(self, servicenow_connector,
                                                      mock_data_entities_processor):
        record = MagicMock()
        record.id = "rec-1"
        mock_data_entities_processor.get_record_by_external_id = AsyncMock(return_value=record)

        await self._run(servicenow_connector, [self._row()])

        mock_data_entities_processor.on_records_deleted_cascade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_removal_pass_reads_rows_the_indexing_query_hides(self, servicenow_connector):
        calls = await self._run(servicenow_connector, [])
        removal = [c for c in calls if c.get("sysparm_fields") == "sys_id,workflow_state,active"]
        assert removal, "the removal pass never ran"
        # A hidden row cannot trigger a removal, so this query must not filter on state.
        assert "workflow_state=published" not in removal[0]["sysparm_query"]

    @pytest.mark.asyncio
    async def test_the_removal_pass_does_not_fetch_article_bodies(self, servicenow_connector):
        calls = await self._run(servicenow_connector, [])
        removal = [c for c in calls if "workflow_state=published" not in c["sysparm_query"]]
        assert removal, "the removal pass never ran"
        # Versioning makes retired rows outnumber published ones, so pulling
        # their bodies would dominate the sync.
        assert "text" not in removal[0]["sysparm_fields"].split(",")


class TestArticleWatermarkAdvance:
    """The indexing query only sees published rows. A sync that removed records
    but indexed nothing therefore left the watermark where it was, and read the
    same window again on every following delta."""

    async def _sync(self, connector, rows, *, fail_removal=False):
        async def _respond(**kwargs):
            fields = kwargs.get("sysparm_fields", "")
            if kwargs.get("tableName") == "sys_audit_delete":
                return _table_api_response([])
            if fields == "sys_id,workflow_state,active":
                if fail_removal:
                    raise ServiceNowAPIError(500, "boom", None)
                return _table_api_response(rows)
            return _table_api_response([])

        connector.article_sync_point = AsyncMock()
        connector.article_sync_point.read_sync_point = AsyncMock(
            return_value={"last_sync_time": "2026-08-27 09:00:00"}
        )
        connector.article_sync_point.update_sync_point = AsyncMock()
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=_respond)
        connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)
        await connector._sync_articles()
        return connector.article_sync_point.update_sync_point

    @pytest.mark.asyncio
    async def test_a_removal_only_sync_still_moves_the_watermark(self, servicenow_connector):
        upd = await self._sync(servicenow_connector, [
            {"sys_id": "art1", "workflow_state": "outdated", "active": "true",
             "sys_updated_on": "2026-08-27 09:38:57"},
        ])
        calls = {c.args[0]: c.args[1] for c in upd.await_args_list}
        assert calls["articles"]["last_sync_time"] == "2026-08-27 09:38:57"

    @pytest.mark.asyncio
    async def test_a_failed_removal_pass_does_not_move_it(self, servicenow_connector):
        """Advancing on a page that was never read would skip the retirements it
        held, and a watermark only moves forward."""
        upd = await self._sync(servicenow_connector, [], fail_removal=True)
        assert "articles" not in {c.args[0] for c in upd.await_args_list}


class TestDeletedRecordGroupRemoval:
    """Deleting a knowledge base or a category leaves its record group behind,
    with its permission edges. Same reasoning as for articles: the row is gone,
    so only sys_audit_delete says it existed."""

    async def _run(self, connector, rows_by_table):
        seen = []

        async def _respond(**kwargs):
            if kwargs.get("tableName") != "sys_audit_delete":
                return _table_api_response([])
            seen.append(kwargs["sysparm_query"])
            for table, rows in rows_by_table.items():
                if f"tablename={table}" in kwargs["sysparm_query"]:
                    return _table_api_response(rows)
            return _table_api_response([])

        connector.article_sync_point = AsyncMock()
        connector.article_sync_point.read_sync_point = AsyncMock(return_value=None)
        connector.article_sync_point.update_sync_point = AsyncMock()
        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=_respond)
        connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)
        removed = await connector._remove_deleted_record_groups()
        return removed, seen

    @pytest.mark.asyncio
    async def test_both_group_tables_are_read(self, servicenow_connector):
        _, seen = await self._run(servicenow_connector, {})
        assert any("tablename=kb_knowledge_base" in q for q in seen)
        assert any("tablename=kb_category" in q for q in seen)

    @pytest.mark.asyncio
    async def test_a_deleted_knowledge_base_loses_its_record_group(
        self, servicenow_connector, mock_data_entities_processor
    ):
        mock_data_entities_processor.on_record_group_deleted = AsyncMock(return_value=True)

        removed, _ = await self._run(servicenow_connector, {
            "kb_knowledge_base": [{"documentkey": "kb1", "sys_created_on": "2026-08-27 12:00:00"}],
        })

        assert removed == 1
        mock_data_entities_processor.on_record_group_deleted.assert_awaited_once_with(
            "kb1", servicenow_connector.connector_id
        )

    @pytest.mark.asyncio
    async def test_a_group_that_was_never_synced_counts_as_nothing(
        self, servicenow_connector, mock_data_entities_processor
    ):
        mock_data_entities_processor.on_record_group_deleted = AsyncMock(return_value=False)

        removed, _ = await self._run(servicenow_connector, {
            "kb_category": [{"documentkey": "cat1", "sys_created_on": "2026-08-27 12:00:00"}],
        })

        assert removed == 0

    @pytest.mark.asyncio
    async def test_each_table_keeps_its_own_sync_point(self, servicenow_connector,
                                                       mock_data_entities_processor):
        mock_data_entities_processor.on_record_group_deleted = AsyncMock(return_value=True)

        await self._run(servicenow_connector, {
            "kb_knowledge_base": [{"documentkey": "kb1", "sys_created_on": "2026-08-27 12:00:00"}],
            "kb_category": [{"documentkey": "cat1", "sys_created_on": "2026-08-27 12:30:00"}],
        })

        keys = [c.args[0] for c in servicenow_connector.article_sync_point.update_sync_point.await_args_list]
        # One watermark shared between two independent reads would let whichever
        # ran second hide the rows the first had not reached.
        assert sorted(keys) == ["category_deletions", "kb_deletions"]


class TestDetachedAttachmentRemoval:
    """An attachment removed from an article that stays published leaves no
    trace anywhere the other passes look: the article row is untouched except
    for its timestamp, and sys_attachment simply stops returning the row."""

    def _attachment(self, sys_id):
        return AttachmentMetadata(
            sys_id=sys_id, file_name=f"{sys_id}.pdf", content_type="application/pdf",
            size_bytes="10", table_sys_id="art1",
            sys_created_on="2026-08-01 10:00:00", sys_updated_on="2026-08-01 10:00:00",
        )

    def _stored(self, external_id, rec_id):
        r = MagicMock()
        r.id = rec_id
        r.external_record_id = external_id
        return r

    @pytest.mark.asyncio
    async def test_an_attachment_that_went_loses_its_record(self, servicenow_connector,
                                                            mock_data_entities_processor):
        mock_data_entities_processor.get_records_by_parent = AsyncMock(return_value=[
            self._stored("att1", "rec-att1"),
            self._stored("att2", "rec-att2"),
        ])
        mock_data_entities_processor.on_records_deleted_cascade = AsyncMock(
            return_value={"deleted_records": ["rec-att2"]}
        )

        deleted = await servicenow_connector._remove_detached_attachments(
            "art1", [self._attachment("att1")]
        )

        assert deleted == 1
        mock_data_entities_processor.on_records_deleted_cascade.assert_awaited_once_with(
            ["rec-att2"], servicenow_connector.connector_id
        )

    @pytest.mark.asyncio
    async def test_an_attachment_still_there_keeps_its_record(self, servicenow_connector,
                                                             mock_data_entities_processor):
        mock_data_entities_processor.get_records_by_parent = AsyncMock(return_value=[
            self._stored("att1", "rec-att1"),
        ])

        deleted = await servicenow_connector._remove_detached_attachments(
            "art1", [self._attachment("att1")]
        )

        assert deleted == 0
        mock_data_entities_processor.on_records_deleted_cascade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_article_with_no_stored_children_deletes_nothing(
        self, servicenow_connector, mock_data_entities_processor
    ):
        mock_data_entities_processor.get_records_by_parent = AsyncMock(return_value=[])

        deleted = await servicenow_connector._remove_detached_attachments("art1", [])

        assert deleted == 0
        mock_data_entities_processor.on_records_deleted_cascade.assert_not_awaited()


class TestDeletedArticleRemoval:
    """A deleted article leaves no row in kb_knowledge, so the unpublished pass
    cannot see it. ServiceNow records the deletion in sys_audit_delete, keyed by
    sys_id in documentkey — the only trace an integration can read.

    Confirmed on a developer instance: an article deleted through the Table API
    appeared there within the second, and its record survived every sync."""

    def _sync_point(self, connector, stored=None):
        connector.article_sync_point = AsyncMock()
        connector.article_sync_point.read_sync_point = AsyncMock(return_value=stored)
        connector.article_sync_point.update_sync_point = AsyncMock()
        return connector.article_sync_point

    async def _run(self, connector, audit_rows, *, error=None, stored=None):
        self._sync_point(connector, stored)

        async def _respond(**kwargs):
            if kwargs.get("tableName") != "sys_audit_delete":
                return _table_api_response([])
            if error is not None:
                raise error
            return _table_api_response(audit_rows)

        mock_ds = AsyncMock()
        mock_ds.get_now_table_tableName = AsyncMock(side_effect=_respond)
        connector._get_fresh_datasource = AsyncMock(return_value=mock_ds)
        connector._captured_ds = mock_ds
        return await connector._remove_deleted_articles()

    @pytest.mark.asyncio
    async def test_a_deleted_article_loses_its_record(self, servicenow_connector,
                                                     mock_data_entities_processor):
        record = MagicMock()
        record.id = "rec-1"
        mock_data_entities_processor.get_record_by_external_id = AsyncMock(return_value=record)
        mock_data_entities_processor.on_records_deleted_cascade = AsyncMock(
            return_value={"deleted_records": ["rec-1"]}
        )

        removed = await self._run(servicenow_connector, [
            {"documentkey": "art1", "sys_created_on": "2026-08-27 09:18:06"},
        ])

        assert removed == 1
        mock_data_entities_processor.on_records_deleted_cascade.assert_awaited_once_with(
            ["rec-1"], servicenow_connector.connector_id
        )

    @pytest.mark.asyncio
    async def test_only_knowledge_deletions_are_read(self, servicenow_connector):
        await self._run(servicenow_connector, [])
        q = servicenow_connector._captured_ds.get_now_table_tableName.call_args.kwargs["sysparm_query"]
        # The table records deletions for every table on the instance, so the
        # filter has to be server-side.
        assert "tablename=kb_knowledge" in q

    @pytest.mark.asyncio
    async def test_it_reads_its_own_sync_point_not_the_articles_one(self, servicenow_connector):
        """The articles watermark is the newest sys_updated_on of a published
        article. Reusing it skips every deletion older than that timestamp, and
        skips it for good, because a watermark only moves forward."""
        await self._run(servicenow_connector, [],
                        stored={"last_sync_time": "2026-08-27 09:18:00"})

        key = servicenow_connector.article_sync_point.read_sync_point.call_args.args[0]
        assert key == "deletions"
        q = servicenow_connector._captured_ds.get_now_table_tableName.call_args.kwargs["sysparm_query"]
        assert "sys_created_on>2026-08-27 09:18:00" in q

    @pytest.mark.asyncio
    async def test_a_clean_pass_advances_its_own_watermark(self, servicenow_connector):
        await self._run(servicenow_connector, [
            {"documentkey": "art1", "sys_created_on": "2026-08-27 09:18:06"},
            {"documentkey": "art2", "sys_created_on": "2026-08-27 10:02:11"},
        ])

        servicenow_connector.article_sync_point.update_sync_point.assert_awaited_once_with(
            "deletions", {"last_sync_time": "2026-08-27 10:02:11"}
        )

    @pytest.mark.asyncio
    async def test_a_failed_pass_leaves_the_watermark_alone(self, servicenow_connector):
        """Advancing past a page that was never read would lose those deletions
        permanently; leaving it puts them in the next pass instead."""
        await self._run(servicenow_connector, [],
                        error=ServiceNowAPIError(403, "forbidden", None))

        servicenow_connector.article_sync_point.update_sync_point.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unreadable_audit_table_does_not_fail_the_sync(
        self, servicenow_connector, mock_data_entities_processor
    ):
        """A least-privilege integration user may not read sys_audit_delete.
        Losing deletion detection beats refusing to sync."""
        removed = await self._run(
            servicenow_connector, [],
            error=ServiceNowAPIError(403, "forbidden", None),
        )

        assert removed == 0
        mock_data_entities_processor.on_records_deleted_cascade.assert_not_awaited()


class TestArticleReadCriteriaOverride:
    """ServiceNow applies article Can Read criteria as an override only when
    glide.knowman.apply_article_read_criteria is true. Confirmed on a developer
    instance: with the property false, a knowledge base grant opens an article
    whose own criteria name a different group."""

    def _article(self):
        return KBKnowledge(
            sys_id="art1",
            short_description="Restricted",
            kb_category="cat1",
            can_read_user_criteria="crit1",
        )

    def test_article_inherits_the_base_grant_by_default(self, servicenow_connector):
        servicenow_connector.apply_article_read_criteria = False

        record = servicenow_connector._transform_to_article_webpage_record(self._article())

        assert record.inherit_permissions is True

    def test_own_criteria_stop_the_inheritance_when_the_property_is_on(self, servicenow_connector):
        servicenow_connector.apply_article_read_criteria = True

        record = servicenow_connector._transform_to_article_webpage_record(self._article())

        assert record.inherit_permissions is False

    def test_an_article_without_criteria_always_inherits(self, servicenow_connector):
        servicenow_connector.apply_article_read_criteria = True
        article = self._article()
        article.can_read_user_criteria = ""

        record = servicenow_connector._transform_to_article_webpage_record(article)

        assert record.inherit_permissions is True

    def test_the_attachment_follows_its_article(self, servicenow_connector):
        record = servicenow_connector._transform_to_attachment_file_record(
            AttachmentMetadata(
                sys_id="att1",
                file_name="doc.pdf",
                content_type="application/pdf",
                size_bytes="1024",
                table_sys_id="art1",
                sys_created_on="2026-08-01 10:00:00",
                sys_updated_on="2026-08-01 10:00:00",
            ),
            parent_record_group_type=RecordGroupType.SERVICENOW_CATEGORY,
            parent_external_record_group_id="cat1",
            inherit_permissions=False,
        )

        assert record.inherit_permissions is False


# ===========================================================================
# Token refresh reaches the transport
# ===========================================================================

class TestAccessTokenRefresh:
    def test_set_access_token_rewrites_the_authorization_header(self):
        """The transport sends headers, so a token written anywhere else is inert."""
        client = ServiceNowRESTClientViaOAuthAuthorizationCode(
            instance_url="https://dev194883.service-now.com",
            client_id="cid",
            client_secret="secret",
            redirect_uri="http://localhost/cb",
            access_token="first-token",
        )
        assert client.headers["Authorization"] == "Bearer first-token"

        client.set_access_token("second-token")

        assert client.access_token == "second-token"
        assert client.headers["Authorization"] == "Bearer second-token"
