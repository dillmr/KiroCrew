import { AlertTriangle, CheckCircle2 } from 'lucide-react'

import type { AcpBackendProbe } from '../../api/client'
import { Badge } from '../ui'
import { i18nT } from '../../i18n/t'

/**
 * One harness's readiness as a badge, in the words first-run setup uses.
 *
 * Shared by the setup gate's "Use other coding agents" picker and Settings >
 * Agent Harness, so a harness reads the same in both places. The precedence is
 * the probe's: an absent binary outranks a cached absence, which outranks a
 * failed check, which outranks a sandbox verdict. `notOffered` is the one state
 * only Settings knows — a harness the build lists but never offers — and it
 * outranks everything, because nothing about installing or re-checking applies
 * to an option that is not on the table.
 */
export function AgentStatusBadge({
  probe,
  blocked,
  notOffered = false,
}: {
  probe: AcpBackendProbe
  /** The host sandbox cannot confine this harness (setup gate only). */
  blocked: boolean
  /** This build never offers the harness (Settings only). */
  notOffered?: boolean
}) {
  if (notOffered) {
    return <Badge variant="muted">{i18nT('pages.developer.agentBackendTab.word_not_offered')}</Badge>
  }
  if (probe.installed === 'missing') {
    return <Badge variant="muted">{i18nT('components.kiroPrerequisiteGate.agent_not_installed')}</Badge>
  }
  if (probe.restart_required) {
    return <Badge variant="warn">{i18nT('components.kiroPrerequisiteGate.agent_restart_needed')}</Badge>
  }
  if (probe.installed === 'unknown') {
    return <Badge variant="muted">{i18nT('components.kiroPrerequisiteGate.agent_unverified')}</Badge>
  }
  if (blocked) {
    return (
      <Badge variant="warn">
        <AlertTriangle className="lucide-inline" /> {i18nT('components.kiroPrerequisiteGate.sandbox_unavailable')}
      </Badge>
    )
  }
  return (
    <Badge variant="ok">
      <CheckCircle2 className="lucide-inline" /> {i18nT('components.kiroPrerequisiteGate.agent_installed')}
    </Badge>
  )
}
