function Lg(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const i in r)if(i!=="default"&&!(i in e)){const a=Object.getOwnPropertyDescriptor(r,i);a&&Object.defineProperty(e,i,a.get?a:{enumerable:!0,get:()=>r[i]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const a of i)if(a.type==="childList")for(const o of a.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&r(o)}).observe(document,{childList:!0,subtree:!0});function n(i){const a={};return i.integrity&&(a.integrity=i.integrity),i.referrerPolicy&&(a.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?a.credentials="include":i.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function r(i){if(i.ep)return;i.ep=!0;const a=n(i);fetch(i.href,a)}})();var Da=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function ho(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var Tp={exports:{}},go={},Lp={exports:{}},J={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Oi=Symbol.for("react.element"),Ag=Symbol.for("react.portal"),Rg=Symbol.for("react.fragment"),jg=Symbol.for("react.strict_mode"),Ig=Symbol.for("react.profiler"),Dg=Symbol.for("react.provider"),Fg=Symbol.for("react.context"),Mg=Symbol.for("react.forward_ref"),Og=Symbol.for("react.suspense"),qg=Symbol.for("react.memo"),Wg=Symbol.for("react.lazy"),lc=Symbol.iterator;function Bg(e){return e===null||typeof e!="object"?null:(e=lc&&e[lc]||e["@@iterator"],typeof e=="function"?e:null)}var Ap={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},Rp=Object.assign,jp={};function Lr(e,t,n){this.props=e,this.context=t,this.refs=jp,this.updater=n||Ap}Lr.prototype.isReactComponent={};Lr.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};Lr.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function Ip(){}Ip.prototype=Lr.prototype;function Hl(e,t,n){this.props=e,this.context=t,this.refs=jp,this.updater=n||Ap}var $l=Hl.prototype=new Ip;$l.constructor=Hl;Rp($l,Lr.prototype);$l.isPureReactComponent=!0;var uc=Array.isArray,Dp=Object.prototype.hasOwnProperty,Ql={current:null},Fp={key:!0,ref:!0,__self:!0,__source:!0};function Mp(e,t,n){var r,i={},a=null,o=null;if(t!=null)for(r in t.ref!==void 0&&(o=t.ref),t.key!==void 0&&(a=""+t.key),t)Dp.call(t,r)&&!Fp.hasOwnProperty(r)&&(i[r]=t[r]);var s=arguments.length-2;if(s===1)i.children=n;else if(1<s){for(var l=Array(s),u=0;u<s;u++)l[u]=arguments[u+2];i.children=l}if(e&&e.defaultProps)for(r in s=e.defaultProps,s)i[r]===void 0&&(i[r]=s[r]);return{$$typeof:Oi,type:e,key:a,ref:o,props:i,_owner:Ql.current}}function Ug(e,t){return{$$typeof:Oi,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function Xl(e){return typeof e=="object"&&e!==null&&e.$$typeof===Oi}function Vg(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var cc=/\/+/g;function Bo(e,t){return typeof e=="object"&&e!==null&&e.key!=null?Vg(""+e.key):t.toString(36)}function ka(e,t,n,r,i){var a=typeof e;(a==="undefined"||a==="boolean")&&(e=null);var o=!1;if(e===null)o=!0;else switch(a){case"string":case"number":o=!0;break;case"object":switch(e.$$typeof){case Oi:case Ag:o=!0}}if(o)return o=e,i=i(o),e=r===""?"."+Bo(o,0):r,uc(i)?(n="",e!=null&&(n=e.replace(cc,"$&/")+"/"),ka(i,t,n,"",function(u){return u})):i!=null&&(Xl(i)&&(i=Ug(i,n+(!i.key||o&&o.key===i.key?"":(""+i.key).replace(cc,"$&/")+"/")+e)),t.push(i)),1;if(o=0,r=r===""?".":r+":",uc(e))for(var s=0;s<e.length;s++){a=e[s];var l=r+Bo(a,s);o+=ka(a,t,n,l,i)}else if(l=Bg(e),typeof l=="function")for(e=l.call(e),s=0;!(a=e.next()).done;)a=a.value,l=r+Bo(a,s++),o+=ka(a,t,n,l,i);else if(a==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return o}function Gi(e,t,n){if(e==null)return e;var r=[],i=0;return ka(e,r,"","",function(a){return t.call(n,a,i++)}),r}function Hg(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var He={current:null},ba={transition:null},$g={ReactCurrentDispatcher:He,ReactCurrentBatchConfig:ba,ReactCurrentOwner:Ql};function Op(){throw Error("act(...) is not supported in production builds of React.")}J.Children={map:Gi,forEach:function(e,t,n){Gi(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return Gi(e,function(){t++}),t},toArray:function(e){return Gi(e,function(t){return t})||[]},only:function(e){if(!Xl(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};J.Component=Lr;J.Fragment=Rg;J.Profiler=Ig;J.PureComponent=Hl;J.StrictMode=jg;J.Suspense=Og;J.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=$g;J.act=Op;J.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=Rp({},e.props),i=e.key,a=e.ref,o=e._owner;if(t!=null){if(t.ref!==void 0&&(a=t.ref,o=Ql.current),t.key!==void 0&&(i=""+t.key),e.type&&e.type.defaultProps)var s=e.type.defaultProps;for(l in t)Dp.call(t,l)&&!Fp.hasOwnProperty(l)&&(r[l]=t[l]===void 0&&s!==void 0?s[l]:t[l])}var l=arguments.length-2;if(l===1)r.children=n;else if(1<l){s=Array(l);for(var u=0;u<l;u++)s[u]=arguments[u+2];r.children=s}return{$$typeof:Oi,type:e.type,key:i,ref:a,props:r,_owner:o}};J.createContext=function(e){return e={$$typeof:Fg,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:Dg,_context:e},e.Consumer=e};J.createElement=Mp;J.createFactory=function(e){var t=Mp.bind(null,e);return t.type=e,t};J.createRef=function(){return{current:null}};J.forwardRef=function(e){return{$$typeof:Mg,render:e}};J.isValidElement=Xl;J.lazy=function(e){return{$$typeof:Wg,_payload:{_status:-1,_result:e},_init:Hg}};J.memo=function(e,t){return{$$typeof:qg,type:e,compare:t===void 0?null:t}};J.startTransition=function(e){var t=ba.transition;ba.transition={};try{e()}finally{ba.transition=t}};J.unstable_act=Op;J.useCallback=function(e,t){return He.current.useCallback(e,t)};J.useContext=function(e){return He.current.useContext(e)};J.useDebugValue=function(){};J.useDeferredValue=function(e){return He.current.useDeferredValue(e)};J.useEffect=function(e,t){return He.current.useEffect(e,t)};J.useId=function(){return He.current.useId()};J.useImperativeHandle=function(e,t,n){return He.current.useImperativeHandle(e,t,n)};J.useInsertionEffect=function(e,t){return He.current.useInsertionEffect(e,t)};J.useLayoutEffect=function(e,t){return He.current.useLayoutEffect(e,t)};J.useMemo=function(e,t){return He.current.useMemo(e,t)};J.useReducer=function(e,t,n){return He.current.useReducer(e,t,n)};J.useRef=function(e){return He.current.useRef(e)};J.useState=function(e){return He.current.useState(e)};J.useSyncExternalStore=function(e,t,n){return He.current.useSyncExternalStore(e,t,n)};J.useTransition=function(){return He.current.useTransition()};J.version="18.3.1";Lp.exports=J;var z=Lp.exports;const Ft=ho(z),Qg=Lg({__proto__:null,default:Ft},[z]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Xg=z,Kg=Symbol.for("react.element"),Yg=Symbol.for("react.fragment"),Gg=Object.prototype.hasOwnProperty,Zg=Xg.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Jg={key:!0,ref:!0,__self:!0,__source:!0};function qp(e,t,n){var r,i={},a=null,o=null;n!==void 0&&(a=""+n),t.key!==void 0&&(a=""+t.key),t.ref!==void 0&&(o=t.ref);for(r in t)Gg.call(t,r)&&!Jg.hasOwnProperty(r)&&(i[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)i[r]===void 0&&(i[r]=t[r]);return{$$typeof:Kg,type:e,key:a,ref:o,props:i,_owner:Zg.current}}go.Fragment=Yg;go.jsx=qp;go.jsxs=qp;Tp.exports=go;var x=Tp.exports,Ds={},Wp={exports:{}},lt={},Bp={exports:{}},Up={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(j,V){var v=j.length;j.push(V);e:for(;0<v;){var B=v-1>>>1,$=j[B];if(0<i($,V))j[B]=V,j[v]=$,v=B;else break e}}function n(j){return j.length===0?null:j[0]}function r(j){if(j.length===0)return null;var V=j[0],v=j.pop();if(v!==V){j[0]=v;e:for(var B=0,$=j.length,w=$>>>1;B<w;){var ie=2*(B+1)-1,ke=j[ie],le=ie+1,ct=j[le];if(0>i(ke,v))le<$&&0>i(ct,ke)?(j[B]=ct,j[le]=v,B=le):(j[B]=ke,j[ie]=v,B=ie);else if(le<$&&0>i(ct,v))j[B]=ct,j[le]=v,B=le;else break e}}return V}function i(j,V){var v=j.sortIndex-V.sortIndex;return v!==0?v:j.id-V.id}if(typeof performance=="object"&&typeof performance.now=="function"){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var l=[],u=[],d=1,c=null,p=3,f=!1,y=!1,_=!1,S=typeof setTimeout=="function"?setTimeout:null,m=typeof clearTimeout=="function"?clearTimeout:null,h=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function g(j){for(var V=n(u);V!==null;){if(V.callback===null)r(u);else if(V.startTime<=j)r(u),V.sortIndex=V.expirationTime,t(l,V);else break;V=n(u)}}function C(j){if(_=!1,g(j),!y)if(n(l)!==null)y=!0,X(E);else{var V=n(u);V!==null&&re(C,V.startTime-j)}}function E(j,V){y=!1,_&&(_=!1,m(T),T=-1),f=!0;var v=p;try{for(g(V),c=n(l);c!==null&&(!(c.expirationTime>V)||j&&!I());){var B=c.callback;if(typeof B=="function"){c.callback=null,p=c.priorityLevel;var $=B(c.expirationTime<=V);V=e.unstable_now(),typeof $=="function"?c.callback=$:c===n(l)&&r(l),g(V)}else r(l);c=n(l)}if(c!==null)var w=!0;else{var ie=n(u);ie!==null&&re(C,ie.startTime-V),w=!1}return w}finally{c=null,p=v,f=!1}}var b=!1,P=null,T=-1,D=5,k=-1;function I(){return!(e.unstable_now()-k<D)}function W(){if(P!==null){var j=e.unstable_now();k=j;var V=!0;try{V=P(!0,j)}finally{V?K():(b=!1,P=null)}}else b=!1}var K;if(typeof h=="function")K=function(){h(W)};else if(typeof MessageChannel<"u"){var te=new MessageChannel,F=te.port2;te.port1.onmessage=W,K=function(){F.postMessage(null)}}else K=function(){S(W,0)};function X(j){P=j,b||(b=!0,K())}function re(j,V){T=S(function(){j(e.unstable_now())},V)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(j){j.callback=null},e.unstable_continueExecution=function(){y||f||(y=!0,X(E))},e.unstable_forceFrameRate=function(j){0>j||125<j?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):D=0<j?Math.floor(1e3/j):5},e.unstable_getCurrentPriorityLevel=function(){return p},e.unstable_getFirstCallbackNode=function(){return n(l)},e.unstable_next=function(j){switch(p){case 1:case 2:case 3:var V=3;break;default:V=p}var v=p;p=V;try{return j()}finally{p=v}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(j,V){switch(j){case 1:case 2:case 3:case 4:case 5:break;default:j=3}var v=p;p=j;try{return V()}finally{p=v}},e.unstable_scheduleCallback=function(j,V,v){var B=e.unstable_now();switch(typeof v=="object"&&v!==null?(v=v.delay,v=typeof v=="number"&&0<v?B+v:B):v=B,j){case 1:var $=-1;break;case 2:$=250;break;case 5:$=1073741823;break;case 4:$=1e4;break;default:$=5e3}return $=v+$,j={id:d++,callback:V,priorityLevel:j,startTime:v,expirationTime:$,sortIndex:-1},v>B?(j.sortIndex=v,t(u,j),n(l)===null&&j===n(u)&&(_?(m(T),T=-1):_=!0,re(C,v-B))):(j.sortIndex=$,t(l,j),y||f||(y=!0,X(E))),j},e.unstable_shouldYield=I,e.unstable_wrapCallback=function(j){var V=p;return function(){var v=p;p=V;try{return j.apply(this,arguments)}finally{p=v}}}})(Up);Bp.exports=Up;var ey=Bp.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var ty=z,st=ey;function A(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var Vp=new Set,hi={};function Xn(e,t){wr(e,t),wr(e+"Capture",t)}function wr(e,t){for(hi[e]=t,e=0;e<t.length;e++)Vp.add(t[e])}var Yt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),Fs=Object.prototype.hasOwnProperty,ny=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,dc={},pc={};function ry(e){return Fs.call(pc,e)?!0:Fs.call(dc,e)?!1:ny.test(e)?pc[e]=!0:(dc[e]=!0,!1)}function iy(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function ay(e,t,n,r){if(t===null||typeof t>"u"||iy(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function $e(e,t,n,r,i,a,o){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=i,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=a,this.removeEmptyString=o}var Re={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){Re[e]=new $e(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];Re[t]=new $e(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){Re[e]=new $e(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){Re[e]=new $e(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){Re[e]=new $e(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){Re[e]=new $e(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){Re[e]=new $e(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){Re[e]=new $e(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){Re[e]=new $e(e,5,!1,e.toLowerCase(),null,!1,!1)});var Kl=/[\-:]([a-z])/g;function Yl(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(Kl,Yl);Re[t]=new $e(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(Kl,Yl);Re[t]=new $e(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(Kl,Yl);Re[t]=new $e(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){Re[e]=new $e(e,1,!1,e.toLowerCase(),null,!1,!1)});Re.xlinkHref=new $e("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){Re[e]=new $e(e,1,!1,e.toLowerCase(),null,!0,!0)});function Gl(e,t,n,r){var i=Re.hasOwnProperty(t)?Re[t]:null;(i!==null?i.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(ay(t,n,i,r)&&(n=null),r||i===null?ry(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):i.mustUseProperty?e[i.propertyName]=n===null?i.type===3?!1:"":n:(t=i.attributeName,r=i.attributeNamespace,n===null?e.removeAttribute(t):(i=i.type,n=i===3||i===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var en=ty.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,Zi=Symbol.for("react.element"),tr=Symbol.for("react.portal"),nr=Symbol.for("react.fragment"),Zl=Symbol.for("react.strict_mode"),Ms=Symbol.for("react.profiler"),Hp=Symbol.for("react.provider"),$p=Symbol.for("react.context"),Jl=Symbol.for("react.forward_ref"),Os=Symbol.for("react.suspense"),qs=Symbol.for("react.suspense_list"),eu=Symbol.for("react.memo"),an=Symbol.for("react.lazy"),Qp=Symbol.for("react.offscreen"),fc=Symbol.iterator;function Or(e){return e===null||typeof e!="object"?null:(e=fc&&e[fc]||e["@@iterator"],typeof e=="function"?e:null)}var xe=Object.assign,Uo;function Yr(e){if(Uo===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);Uo=t&&t[1]||""}return`
`+Uo+e}var Vo=!1;function Ho(e,t){if(!e||Vo)return"";Vo=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var r=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){r=u}e.call(t.prototype)}else{try{throw Error()}catch(u){r=u}e()}}catch(u){if(u&&r&&typeof u.stack=="string"){for(var i=u.stack.split(`
`),a=r.stack.split(`
`),o=i.length-1,s=a.length-1;1<=o&&0<=s&&i[o]!==a[s];)s--;for(;1<=o&&0<=s;o--,s--)if(i[o]!==a[s]){if(o!==1||s!==1)do if(o--,s--,0>s||i[o]!==a[s]){var l=`
`+i[o].replace(" at new "," at ");return e.displayName&&l.includes("<anonymous>")&&(l=l.replace("<anonymous>",e.displayName)),l}while(1<=o&&0<=s);break}}}finally{Vo=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?Yr(e):""}function oy(e){switch(e.tag){case 5:return Yr(e.type);case 16:return Yr("Lazy");case 13:return Yr("Suspense");case 19:return Yr("SuspenseList");case 0:case 2:case 15:return e=Ho(e.type,!1),e;case 11:return e=Ho(e.type.render,!1),e;case 1:return e=Ho(e.type,!0),e;default:return""}}function Ws(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case nr:return"Fragment";case tr:return"Portal";case Ms:return"Profiler";case Zl:return"StrictMode";case Os:return"Suspense";case qs:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case $p:return(e.displayName||"Context")+".Consumer";case Hp:return(e._context.displayName||"Context")+".Provider";case Jl:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case eu:return t=e.displayName||null,t!==null?t:Ws(e.type)||"Memo";case an:t=e._payload,e=e._init;try{return Ws(e(t))}catch{}}return null}function sy(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return Ws(t);case 8:return t===Zl?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function kn(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function Xp(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function ly(e){var t=Xp(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var i=n.get,a=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(o){r=""+o,a.call(this,o)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(o){r=""+o},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function Ji(e){e._valueTracker||(e._valueTracker=ly(e))}function Kp(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=Xp(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function Fa(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function Bs(e,t){var n=t.checked;return xe({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function mc(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=kn(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Yp(e,t){t=t.checked,t!=null&&Gl(e,"checked",t,!1)}function Us(e,t){Yp(e,t);var n=kn(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?Vs(e,t.type,n):t.hasOwnProperty("defaultValue")&&Vs(e,t.type,kn(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function hc(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function Vs(e,t,n){(t!=="number"||Fa(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var Gr=Array.isArray;function mr(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t["$"+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty("$"+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=""+kn(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function Hs(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(A(91));return xe({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function gc(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(A(92));if(Gr(n)){if(1<n.length)throw Error(A(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:kn(n)}}function Gp(e,t){var n=kn(t.value),r=kn(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function yc(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Zp(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function $s(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Zp(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var ea,Jp=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,i){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,i)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(ea=ea||document.createElement("div"),ea.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=ea.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function gi(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var ni={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},uy=["Webkit","ms","Moz","O"];Object.keys(ni).forEach(function(e){uy.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),ni[t]=ni[e]})});function ef(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||ni.hasOwnProperty(e)&&ni[e]?(""+t).trim():t+"px"}function tf(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,i=ef(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,i):e[n]=i}}var cy=xe({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Qs(e,t){if(t){if(cy[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(A(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(A(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(A(61))}if(t.style!=null&&typeof t.style!="object")throw Error(A(62))}}function Xs(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Ks=null;function tu(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Ys=null,hr=null,gr=null;function xc(e){if(e=Bi(e)){if(typeof Ys!="function")throw Error(A(280));var t=e.stateNode;t&&(t=wo(t),Ys(e.stateNode,e.type,t))}}function nf(e){hr?gr?gr.push(e):gr=[e]:hr=e}function rf(){if(hr){var e=hr,t=gr;if(gr=hr=null,xc(e),t)for(e=0;e<t.length;e++)xc(t[e])}}function af(e,t){return e(t)}function of(){}var $o=!1;function sf(e,t,n){if($o)return e(t,n);$o=!0;try{return af(e,t,n)}finally{$o=!1,(hr!==null||gr!==null)&&(of(),rf())}}function yi(e,t){var n=e.stateNode;if(n===null)return null;var r=wo(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(A(231,t,typeof n));return n}var Gs=!1;if(Yt)try{var qr={};Object.defineProperty(qr,"passive",{get:function(){Gs=!0}}),window.addEventListener("test",qr,qr),window.removeEventListener("test",qr,qr)}catch{Gs=!1}function dy(e,t,n,r,i,a,o,s,l){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(d){this.onError(d)}}var ri=!1,Ma=null,Oa=!1,Zs=null,py={onError:function(e){ri=!0,Ma=e}};function fy(e,t,n,r,i,a,o,s,l){ri=!1,Ma=null,dy.apply(py,arguments)}function my(e,t,n,r,i,a,o,s,l){if(fy.apply(this,arguments),ri){if(ri){var u=Ma;ri=!1,Ma=null}else throw Error(A(198));Oa||(Oa=!0,Zs=u)}}function Kn(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function lf(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function _c(e){if(Kn(e)!==e)throw Error(A(188))}function hy(e){var t=e.alternate;if(!t){if(t=Kn(e),t===null)throw Error(A(188));return t!==e?null:e}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var a=i.alternate;if(a===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===a.child){for(a=i.child;a;){if(a===n)return _c(i),e;if(a===r)return _c(i),t;a=a.sibling}throw Error(A(188))}if(n.return!==r.return)n=i,r=a;else{for(var o=!1,s=i.child;s;){if(s===n){o=!0,n=i,r=a;break}if(s===r){o=!0,r=i,n=a;break}s=s.sibling}if(!o){for(s=a.child;s;){if(s===n){o=!0,n=a,r=i;break}if(s===r){o=!0,r=a,n=i;break}s=s.sibling}if(!o)throw Error(A(189))}}if(n.alternate!==r)throw Error(A(190))}if(n.tag!==3)throw Error(A(188));return n.stateNode.current===n?e:t}function uf(e){return e=hy(e),e!==null?cf(e):null}function cf(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=cf(e);if(t!==null)return t;e=e.sibling}return null}var df=st.unstable_scheduleCallback,vc=st.unstable_cancelCallback,gy=st.unstable_shouldYield,yy=st.unstable_requestPaint,be=st.unstable_now,xy=st.unstable_getCurrentPriorityLevel,nu=st.unstable_ImmediatePriority,pf=st.unstable_UserBlockingPriority,qa=st.unstable_NormalPriority,_y=st.unstable_LowPriority,ff=st.unstable_IdlePriority,yo=null,Mt=null;function vy(e){if(Mt&&typeof Mt.onCommitFiberRoot=="function")try{Mt.onCommitFiberRoot(yo,e,void 0,(e.current.flags&128)===128)}catch{}}var Pt=Math.clz32?Math.clz32:by,wy=Math.log,ky=Math.LN2;function by(e){return e>>>=0,e===0?32:31-(wy(e)/ky|0)|0}var ta=64,na=4194304;function Zr(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function Wa(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,i=e.suspendedLanes,a=e.pingedLanes,o=n&268435455;if(o!==0){var s=o&~i;s!==0?r=Zr(s):(a&=o,a!==0&&(r=Zr(a)))}else o=n&~i,o!==0?r=Zr(o):a!==0&&(r=Zr(a));if(r===0)return 0;if(t!==0&&t!==r&&!(t&i)&&(i=r&-r,a=t&-t,i>=a||i===16&&(a&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-Pt(t),i=1<<n,r|=e[n],t&=~i;return r}function Sy(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Cy(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes;0<a;){var o=31-Pt(a),s=1<<o,l=i[o];l===-1?(!(s&n)||s&r)&&(i[o]=Sy(s,t)):l<=t&&(e.expiredLanes|=s),a&=~s}}function Js(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function mf(){var e=ta;return ta<<=1,!(ta&4194240)&&(ta=64),e}function Qo(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function qi(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-Pt(t),e[t]=n}function zy(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var i=31-Pt(n),a=1<<i;t[i]=0,r[i]=-1,e[i]=-1,n&=~a}}function ru(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Pt(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}var se=0;function hf(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var gf,iu,yf,xf,_f,el=!1,ra=[],pn=null,fn=null,mn=null,xi=new Map,_i=new Map,sn=[],Ey="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function wc(e,t){switch(e){case"focusin":case"focusout":pn=null;break;case"dragenter":case"dragleave":fn=null;break;case"mouseover":case"mouseout":mn=null;break;case"pointerover":case"pointerout":xi.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":_i.delete(t.pointerId)}}function Wr(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Bi(t),t!==null&&iu(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Py(e,t,n,r,i){switch(t){case"focusin":return pn=Wr(pn,e,t,n,r,i),!0;case"dragenter":return fn=Wr(fn,e,t,n,r,i),!0;case"mouseover":return mn=Wr(mn,e,t,n,r,i),!0;case"pointerover":var a=i.pointerId;return xi.set(a,Wr(xi.get(a)||null,e,t,n,r,i)),!0;case"gotpointercapture":return a=i.pointerId,_i.set(a,Wr(_i.get(a)||null,e,t,n,r,i)),!0}return!1}function vf(e){var t=Fn(e.target);if(t!==null){var n=Kn(t);if(n!==null){if(t=n.tag,t===13){if(t=lf(n),t!==null){e.blockedOn=t,_f(e.priority,function(){yf(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Sa(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=tl(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);Ks=r,n.target.dispatchEvent(r),Ks=null}else return t=Bi(n),t!==null&&iu(t),e.blockedOn=n,!1;t.shift()}return!0}function kc(e,t,n){Sa(e)&&n.delete(t)}function Ny(){el=!1,pn!==null&&Sa(pn)&&(pn=null),fn!==null&&Sa(fn)&&(fn=null),mn!==null&&Sa(mn)&&(mn=null),xi.forEach(kc),_i.forEach(kc)}function Br(e,t){e.blockedOn===t&&(e.blockedOn=null,el||(el=!0,st.unstable_scheduleCallback(st.unstable_NormalPriority,Ny)))}function vi(e){function t(i){return Br(i,e)}if(0<ra.length){Br(ra[0],e);for(var n=1;n<ra.length;n++){var r=ra[n];r.blockedOn===e&&(r.blockedOn=null)}}for(pn!==null&&Br(pn,e),fn!==null&&Br(fn,e),mn!==null&&Br(mn,e),xi.forEach(t),_i.forEach(t),n=0;n<sn.length;n++)r=sn[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<sn.length&&(n=sn[0],n.blockedOn===null);)vf(n),n.blockedOn===null&&sn.shift()}var yr=en.ReactCurrentBatchConfig,Ba=!0;function Ty(e,t,n,r){var i=se,a=yr.transition;yr.transition=null;try{se=1,au(e,t,n,r)}finally{se=i,yr.transition=a}}function Ly(e,t,n,r){var i=se,a=yr.transition;yr.transition=null;try{se=4,au(e,t,n,r)}finally{se=i,yr.transition=a}}function au(e,t,n,r){if(Ba){var i=tl(e,t,n,r);if(i===null)rs(e,t,r,Ua,n),wc(e,r);else if(Py(i,e,t,n,r))r.stopPropagation();else if(wc(e,r),t&4&&-1<Ey.indexOf(e)){for(;i!==null;){var a=Bi(i);if(a!==null&&gf(a),a=tl(e,t,n,r),a===null&&rs(e,t,r,Ua,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else rs(e,t,r,null,n)}}var Ua=null;function tl(e,t,n,r){if(Ua=null,e=tu(r),e=Fn(e),e!==null)if(t=Kn(e),t===null)e=null;else if(n=t.tag,n===13){if(e=lf(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return Ua=e,null}function wf(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(xy()){case nu:return 1;case pf:return 4;case qa:case _y:return 16;case ff:return 536870912;default:return 16}default:return 16}}var un=null,ou=null,Ca=null;function kf(){if(Ca)return Ca;var e,t=ou,n=t.length,r,i="value"in un?un.value:un.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return Ca=i.slice(e,1<r?1-r:void 0)}function za(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function ia(){return!0}function bc(){return!1}function ut(e){function t(n,r,i,a,o){this._reactName=n,this._targetInst=i,this.type=r,this.nativeEvent=a,this.target=o,this.currentTarget=null;for(var s in e)e.hasOwnProperty(s)&&(n=e[s],this[s]=n?n(a):a[s]);return this.isDefaultPrevented=(a.defaultPrevented!=null?a.defaultPrevented:a.returnValue===!1)?ia:bc,this.isPropagationStopped=bc,this}return xe(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=ia)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=ia)},persist:function(){},isPersistent:ia}),t}var Ar={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},su=ut(Ar),Wi=xe({},Ar,{view:0,detail:0}),Ay=ut(Wi),Xo,Ko,Ur,xo=xe({},Wi,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:lu,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==Ur&&(Ur&&e.type==="mousemove"?(Xo=e.screenX-Ur.screenX,Ko=e.screenY-Ur.screenY):Ko=Xo=0,Ur=e),Xo)},movementY:function(e){return"movementY"in e?e.movementY:Ko}}),Sc=ut(xo),Ry=xe({},xo,{dataTransfer:0}),jy=ut(Ry),Iy=xe({},Wi,{relatedTarget:0}),Yo=ut(Iy),Dy=xe({},Ar,{animationName:0,elapsedTime:0,pseudoElement:0}),Fy=ut(Dy),My=xe({},Ar,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),Oy=ut(My),qy=xe({},Ar,{data:0}),Cc=ut(qy),Wy={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},By={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},Uy={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function Vy(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=Uy[e])?!!t[e]:!1}function lu(){return Vy}var Hy=xe({},Wi,{key:function(e){if(e.key){var t=Wy[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=za(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?By[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:lu,charCode:function(e){return e.type==="keypress"?za(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?za(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),$y=ut(Hy),Qy=xe({},xo,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),zc=ut(Qy),Xy=xe({},Wi,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:lu}),Ky=ut(Xy),Yy=xe({},Ar,{propertyName:0,elapsedTime:0,pseudoElement:0}),Gy=ut(Yy),Zy=xe({},xo,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),Jy=ut(Zy),ex=[9,13,27,32],uu=Yt&&"CompositionEvent"in window,ii=null;Yt&&"documentMode"in document&&(ii=document.documentMode);var tx=Yt&&"TextEvent"in window&&!ii,bf=Yt&&(!uu||ii&&8<ii&&11>=ii),Ec=" ",Pc=!1;function Sf(e,t){switch(e){case"keyup":return ex.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function Cf(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var rr=!1;function nx(e,t){switch(e){case"compositionend":return Cf(t);case"keypress":return t.which!==32?null:(Pc=!0,Ec);case"textInput":return e=t.data,e===Ec&&Pc?null:e;default:return null}}function rx(e,t){if(rr)return e==="compositionend"||!uu&&Sf(e,t)?(e=kf(),Ca=ou=un=null,rr=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return bf&&t.locale!=="ko"?null:t.data;default:return null}}var ix={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Nc(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!ix[e.type]:t==="textarea"}function zf(e,t,n,r){nf(r),t=Va(t,"onChange"),0<t.length&&(n=new su("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var ai=null,wi=null;function ax(e){Ff(e,0)}function _o(e){var t=or(e);if(Kp(t))return e}function ox(e,t){if(e==="change")return t}var Ef=!1;if(Yt){var Go;if(Yt){var Zo="oninput"in document;if(!Zo){var Tc=document.createElement("div");Tc.setAttribute("oninput","return;"),Zo=typeof Tc.oninput=="function"}Go=Zo}else Go=!1;Ef=Go&&(!document.documentMode||9<document.documentMode)}function Lc(){ai&&(ai.detachEvent("onpropertychange",Pf),wi=ai=null)}function Pf(e){if(e.propertyName==="value"&&_o(wi)){var t=[];zf(t,wi,e,tu(e)),sf(ax,t)}}function sx(e,t,n){e==="focusin"?(Lc(),ai=t,wi=n,ai.attachEvent("onpropertychange",Pf)):e==="focusout"&&Lc()}function lx(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return _o(wi)}function ux(e,t){if(e==="click")return _o(t)}function cx(e,t){if(e==="input"||e==="change")return _o(t)}function dx(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var Lt=typeof Object.is=="function"?Object.is:dx;function ki(e,t){if(Lt(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!Fs.call(t,i)||!Lt(e[i],t[i]))return!1}return!0}function Ac(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Rc(e,t){var n=Ac(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=Ac(n)}}function Nf(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Nf(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function Tf(){for(var e=window,t=Fa();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=Fa(e.document)}return t}function cu(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function px(e){var t=Tf(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&Nf(n.ownerDocument.documentElement,n)){if(r!==null&&cu(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var i=n.textContent.length,a=Math.min(r.start,i);r=r.end===void 0?a:Math.min(r.end,i),!e.extend&&a>r&&(i=r,r=a,a=i),i=Rc(n,a);var o=Rc(n,r);i&&o&&(e.rangeCount!==1||e.anchorNode!==i.node||e.anchorOffset!==i.offset||e.focusNode!==o.node||e.focusOffset!==o.offset)&&(t=t.createRange(),t.setStart(i.node,i.offset),e.removeAllRanges(),a>r?(e.addRange(t),e.extend(o.node,o.offset)):(t.setEnd(o.node,o.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var fx=Yt&&"documentMode"in document&&11>=document.documentMode,ir=null,nl=null,oi=null,rl=!1;function jc(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;rl||ir==null||ir!==Fa(r)||(r=ir,"selectionStart"in r&&cu(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),oi&&ki(oi,r)||(oi=r,r=Va(nl,"onSelect"),0<r.length&&(t=new su("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=ir)))}function aa(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var ar={animationend:aa("Animation","AnimationEnd"),animationiteration:aa("Animation","AnimationIteration"),animationstart:aa("Animation","AnimationStart"),transitionend:aa("Transition","TransitionEnd")},Jo={},Lf={};Yt&&(Lf=document.createElement("div").style,"AnimationEvent"in window||(delete ar.animationend.animation,delete ar.animationiteration.animation,delete ar.animationstart.animation),"TransitionEvent"in window||delete ar.transitionend.transition);function vo(e){if(Jo[e])return Jo[e];if(!ar[e])return e;var t=ar[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in Lf)return Jo[e]=t[n];return e}var Af=vo("animationend"),Rf=vo("animationiteration"),jf=vo("animationstart"),If=vo("transitionend"),Df=new Map,Ic="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function Sn(e,t){Df.set(e,t),Xn(t,[e])}for(var es=0;es<Ic.length;es++){var ts=Ic[es],mx=ts.toLowerCase(),hx=ts[0].toUpperCase()+ts.slice(1);Sn(mx,"on"+hx)}Sn(Af,"onAnimationEnd");Sn(Rf,"onAnimationIteration");Sn(jf,"onAnimationStart");Sn("dblclick","onDoubleClick");Sn("focusin","onFocus");Sn("focusout","onBlur");Sn(If,"onTransitionEnd");wr("onMouseEnter",["mouseout","mouseover"]);wr("onMouseLeave",["mouseout","mouseover"]);wr("onPointerEnter",["pointerout","pointerover"]);wr("onPointerLeave",["pointerout","pointerover"]);Xn("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));Xn("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));Xn("onBeforeInput",["compositionend","keypress","textInput","paste"]);Xn("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));Xn("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));Xn("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Jr="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),gx=new Set("cancel close invalid load scroll toggle".split(" ").concat(Jr));function Dc(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,my(r,t,void 0,e),e.currentTarget=null}function Ff(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;e:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],l=s.instance,u=s.currentTarget;if(s=s.listener,l!==a&&i.isPropagationStopped())break e;Dc(i,s,u),a=l}else for(o=0;o<r.length;o++){if(s=r[o],l=s.instance,u=s.currentTarget,s=s.listener,l!==a&&i.isPropagationStopped())break e;Dc(i,s,u),a=l}}}if(Oa)throw e=Zs,Oa=!1,Zs=null,e}function fe(e,t){var n=t[ll];n===void 0&&(n=t[ll]=new Set);var r=e+"__bubble";n.has(r)||(Mf(t,e,2,!1),n.add(r))}function ns(e,t,n){var r=0;t&&(r|=4),Mf(n,e,r,t)}var oa="_reactListening"+Math.random().toString(36).slice(2);function bi(e){if(!e[oa]){e[oa]=!0,Vp.forEach(function(n){n!=="selectionchange"&&(gx.has(n)||ns(n,!1,e),ns(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[oa]||(t[oa]=!0,ns("selectionchange",!1,t))}}function Mf(e,t,n,r){switch(wf(t)){case 1:var i=Ty;break;case 4:i=Ly;break;default:i=au}n=i.bind(null,t,n,e),i=void 0,!Gs||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(i=!0),r?i!==void 0?e.addEventListener(t,n,{capture:!0,passive:i}):e.addEventListener(t,n,!0):i!==void 0?e.addEventListener(t,n,{passive:i}):e.addEventListener(t,n,!1)}function rs(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var o=r.tag;if(o===3||o===4){var s=r.stateNode.containerInfo;if(s===i||s.nodeType===8&&s.parentNode===i)break;if(o===4)for(o=r.return;o!==null;){var l=o.tag;if((l===3||l===4)&&(l=o.stateNode.containerInfo,l===i||l.nodeType===8&&l.parentNode===i))return;o=o.return}for(;s!==null;){if(o=Fn(s),o===null)return;if(l=o.tag,l===5||l===6){r=a=o;continue e}s=s.parentNode}}r=r.return}sf(function(){var u=a,d=tu(n),c=[];e:{var p=Df.get(e);if(p!==void 0){var f=su,y=e;switch(e){case"keypress":if(za(n)===0)break e;case"keydown":case"keyup":f=$y;break;case"focusin":y="focus",f=Yo;break;case"focusout":y="blur",f=Yo;break;case"beforeblur":case"afterblur":f=Yo;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":f=Sc;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":f=jy;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":f=Ky;break;case Af:case Rf:case jf:f=Fy;break;case If:f=Gy;break;case"scroll":f=Ay;break;case"wheel":f=Jy;break;case"copy":case"cut":case"paste":f=Oy;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":f=zc}var _=(t&4)!==0,S=!_&&e==="scroll",m=_?p!==null?p+"Capture":null:p;_=[];for(var h=u,g;h!==null;){g=h;var C=g.stateNode;if(g.tag===5&&C!==null&&(g=C,m!==null&&(C=yi(h,m),C!=null&&_.push(Si(h,C,g)))),S)break;h=h.return}0<_.length&&(p=new f(p,y,null,n,d),c.push({event:p,listeners:_}))}}if(!(t&7)){e:{if(p=e==="mouseover"||e==="pointerover",f=e==="mouseout"||e==="pointerout",p&&n!==Ks&&(y=n.relatedTarget||n.fromElement)&&(Fn(y)||y[Gt]))break e;if((f||p)&&(p=d.window===d?d:(p=d.ownerDocument)?p.defaultView||p.parentWindow:window,f?(y=n.relatedTarget||n.toElement,f=u,y=y?Fn(y):null,y!==null&&(S=Kn(y),y!==S||y.tag!==5&&y.tag!==6)&&(y=null)):(f=null,y=u),f!==y)){if(_=Sc,C="onMouseLeave",m="onMouseEnter",h="mouse",(e==="pointerout"||e==="pointerover")&&(_=zc,C="onPointerLeave",m="onPointerEnter",h="pointer"),S=f==null?p:or(f),g=y==null?p:or(y),p=new _(C,h+"leave",f,n,d),p.target=S,p.relatedTarget=g,C=null,Fn(d)===u&&(_=new _(m,h+"enter",y,n,d),_.target=g,_.relatedTarget=S,C=_),S=C,f&&y)t:{for(_=f,m=y,h=0,g=_;g;g=Zn(g))h++;for(g=0,C=m;C;C=Zn(C))g++;for(;0<h-g;)_=Zn(_),h--;for(;0<g-h;)m=Zn(m),g--;for(;h--;){if(_===m||m!==null&&_===m.alternate)break t;_=Zn(_),m=Zn(m)}_=null}else _=null;f!==null&&Fc(c,p,f,_,!1),y!==null&&S!==null&&Fc(c,S,y,_,!0)}}e:{if(p=u?or(u):window,f=p.nodeName&&p.nodeName.toLowerCase(),f==="select"||f==="input"&&p.type==="file")var E=ox;else if(Nc(p))if(Ef)E=cx;else{E=lx;var b=sx}else(f=p.nodeName)&&f.toLowerCase()==="input"&&(p.type==="checkbox"||p.type==="radio")&&(E=ux);if(E&&(E=E(e,u))){zf(c,E,n,d);break e}b&&b(e,p,u),e==="focusout"&&(b=p._wrapperState)&&b.controlled&&p.type==="number"&&Vs(p,"number",p.value)}switch(b=u?or(u):window,e){case"focusin":(Nc(b)||b.contentEditable==="true")&&(ir=b,nl=u,oi=null);break;case"focusout":oi=nl=ir=null;break;case"mousedown":rl=!0;break;case"contextmenu":case"mouseup":case"dragend":rl=!1,jc(c,n,d);break;case"selectionchange":if(fx)break;case"keydown":case"keyup":jc(c,n,d)}var P;if(uu)e:{switch(e){case"compositionstart":var T="onCompositionStart";break e;case"compositionend":T="onCompositionEnd";break e;case"compositionupdate":T="onCompositionUpdate";break e}T=void 0}else rr?Sf(e,n)&&(T="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(T="onCompositionStart");T&&(bf&&n.locale!=="ko"&&(rr||T!=="onCompositionStart"?T==="onCompositionEnd"&&rr&&(P=kf()):(un=d,ou="value"in un?un.value:un.textContent,rr=!0)),b=Va(u,T),0<b.length&&(T=new Cc(T,e,null,n,d),c.push({event:T,listeners:b}),P?T.data=P:(P=Cf(n),P!==null&&(T.data=P)))),(P=tx?nx(e,n):rx(e,n))&&(u=Va(u,"onBeforeInput"),0<u.length&&(d=new Cc("onBeforeInput","beforeinput",null,n,d),c.push({event:d,listeners:u}),d.data=P))}Ff(c,t)})}function Si(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Va(e,t){for(var n=t+"Capture",r=[];e!==null;){var i=e,a=i.stateNode;i.tag===5&&a!==null&&(i=a,a=yi(e,n),a!=null&&r.unshift(Si(e,a,i)),a=yi(e,t),a!=null&&r.push(Si(e,a,i))),e=e.return}return r}function Zn(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function Fc(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,l=s.alternate,u=s.stateNode;if(l!==null&&l===r)break;s.tag===5&&u!==null&&(s=u,i?(l=yi(n,a),l!=null&&o.unshift(Si(n,l,s))):i||(l=yi(n,a),l!=null&&o.push(Si(n,l,s)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var yx=/\r\n?/g,xx=/\u0000|\uFFFD/g;function Mc(e){return(typeof e=="string"?e:""+e).replace(yx,`
`).replace(xx,"")}function sa(e,t,n){if(t=Mc(t),Mc(e)!==t&&n)throw Error(A(425))}function Ha(){}var il=null,al=null;function ol(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var sl=typeof setTimeout=="function"?setTimeout:void 0,_x=typeof clearTimeout=="function"?clearTimeout:void 0,Oc=typeof Promise=="function"?Promise:void 0,vx=typeof queueMicrotask=="function"?queueMicrotask:typeof Oc<"u"?function(e){return Oc.resolve(null).then(e).catch(wx)}:sl;function wx(e){setTimeout(function(){throw e})}function is(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(r===0){e.removeChild(i),vi(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=i}while(n);vi(t)}function hn(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function qc(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Rr=Math.random().toString(36).slice(2),Dt="__reactFiber$"+Rr,Ci="__reactProps$"+Rr,Gt="__reactContainer$"+Rr,ll="__reactEvents$"+Rr,kx="__reactListeners$"+Rr,bx="__reactHandles$"+Rr;function Fn(e){var t=e[Dt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[Gt]||n[Dt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=qc(e);e!==null;){if(n=e[Dt])return n;e=qc(e)}return t}e=n,n=e.parentNode}return null}function Bi(e){return e=e[Dt]||e[Gt],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function or(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(A(33))}function wo(e){return e[Ci]||null}var ul=[],sr=-1;function Cn(e){return{current:e}}function me(e){0>sr||(e.current=ul[sr],ul[sr]=null,sr--)}function pe(e,t){sr++,ul[sr]=e.current,e.current=t}var bn={},Oe=Cn(bn),Ye=Cn(!1),Bn=bn;function kr(e,t){var n=e.type.contextTypes;if(!n)return bn;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var i={},a;for(a in n)i[a]=t[a];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=i),i}function Ge(e){return e=e.childContextTypes,e!=null}function $a(){me(Ye),me(Oe)}function Wc(e,t,n){if(Oe.current!==bn)throw Error(A(168));pe(Oe,t),pe(Ye,n)}function Of(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var i in r)if(!(i in t))throw Error(A(108,sy(e)||"Unknown",i));return xe({},n,r)}function Qa(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||bn,Bn=Oe.current,pe(Oe,e),pe(Ye,Ye.current),!0}function Bc(e,t,n){var r=e.stateNode;if(!r)throw Error(A(169));n?(e=Of(e,t,Bn),r.__reactInternalMemoizedMergedChildContext=e,me(Ye),me(Oe),pe(Oe,e)):me(Ye),pe(Ye,n)}var $t=null,ko=!1,as=!1;function qf(e){$t===null?$t=[e]:$t.push(e)}function Sx(e){ko=!0,qf(e)}function zn(){if(!as&&$t!==null){as=!0;var e=0,t=se;try{var n=$t;for(se=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}$t=null,ko=!1}catch(i){throw $t!==null&&($t=$t.slice(e+1)),df(nu,zn),i}finally{se=t,as=!1}}return null}var lr=[],ur=0,Xa=null,Ka=0,pt=[],ft=0,Un=null,Qt=1,Xt="";function jn(e,t){lr[ur++]=Ka,lr[ur++]=Xa,Xa=e,Ka=t}function Wf(e,t,n){pt[ft++]=Qt,pt[ft++]=Xt,pt[ft++]=Un,Un=e;var r=Qt;e=Xt;var i=32-Pt(r)-1;r&=~(1<<i),n+=1;var a=32-Pt(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Qt=1<<32-Pt(t)+i|n<<i|r,Xt=a+e}else Qt=1<<a|n<<i|r,Xt=e}function du(e){e.return!==null&&(jn(e,1),Wf(e,1,0))}function pu(e){for(;e===Xa;)Xa=lr[--ur],lr[ur]=null,Ka=lr[--ur],lr[ur]=null;for(;e===Un;)Un=pt[--ft],pt[ft]=null,Xt=pt[--ft],pt[ft]=null,Qt=pt[--ft],pt[ft]=null}var at=null,it=null,he=!1,Et=null;function Bf(e,t){var n=ht(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function Uc(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,at=e,it=hn(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,at=e,it=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=Un!==null?{id:Qt,overflow:Xt}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=ht(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,at=e,it=null,!0):!1;default:return!1}}function cl(e){return(e.mode&1)!==0&&(e.flags&128)===0}function dl(e){if(he){var t=it;if(t){var n=t;if(!Uc(e,t)){if(cl(e))throw Error(A(418));t=hn(n.nextSibling);var r=at;t&&Uc(e,t)?Bf(r,n):(e.flags=e.flags&-4097|2,he=!1,at=e)}}else{if(cl(e))throw Error(A(418));e.flags=e.flags&-4097|2,he=!1,at=e}}}function Vc(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;at=e}function la(e){if(e!==at)return!1;if(!he)return Vc(e),he=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!ol(e.type,e.memoizedProps)),t&&(t=it)){if(cl(e))throw Uf(),Error(A(418));for(;t;)Bf(e,t),t=hn(t.nextSibling)}if(Vc(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(A(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){it=hn(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}it=null}}else it=at?hn(e.stateNode.nextSibling):null;return!0}function Uf(){for(var e=it;e;)e=hn(e.nextSibling)}function br(){it=at=null,he=!1}function fu(e){Et===null?Et=[e]:Et.push(e)}var Cx=en.ReactCurrentBatchConfig;function Vr(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(A(309));var r=n.stateNode}if(!r)throw Error(A(147,e));var i=r,a=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===a?t.ref:(t=function(o){var s=i.refs;o===null?delete s[a]:s[a]=o},t._stringRef=a,t)}if(typeof e!="string")throw Error(A(284));if(!n._owner)throw Error(A(290,e))}return e}function ua(e,t){throw e=Object.prototype.toString.call(t),Error(A(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function Hc(e){var t=e._init;return t(e._payload)}function Vf(e){function t(m,h){if(e){var g=m.deletions;g===null?(m.deletions=[h],m.flags|=16):g.push(h)}}function n(m,h){if(!e)return null;for(;h!==null;)t(m,h),h=h.sibling;return null}function r(m,h){for(m=new Map;h!==null;)h.key!==null?m.set(h.key,h):m.set(h.index,h),h=h.sibling;return m}function i(m,h){return m=_n(m,h),m.index=0,m.sibling=null,m}function a(m,h,g){return m.index=g,e?(g=m.alternate,g!==null?(g=g.index,g<h?(m.flags|=2,h):g):(m.flags|=2,h)):(m.flags|=1048576,h)}function o(m){return e&&m.alternate===null&&(m.flags|=2),m}function s(m,h,g,C){return h===null||h.tag!==6?(h=ps(g,m.mode,C),h.return=m,h):(h=i(h,g),h.return=m,h)}function l(m,h,g,C){var E=g.type;return E===nr?d(m,h,g.props.children,C,g.key):h!==null&&(h.elementType===E||typeof E=="object"&&E!==null&&E.$$typeof===an&&Hc(E)===h.type)?(C=i(h,g.props),C.ref=Vr(m,h,g),C.return=m,C):(C=Ra(g.type,g.key,g.props,null,m.mode,C),C.ref=Vr(m,h,g),C.return=m,C)}function u(m,h,g,C){return h===null||h.tag!==4||h.stateNode.containerInfo!==g.containerInfo||h.stateNode.implementation!==g.implementation?(h=fs(g,m.mode,C),h.return=m,h):(h=i(h,g.children||[]),h.return=m,h)}function d(m,h,g,C,E){return h===null||h.tag!==7?(h=Wn(g,m.mode,C,E),h.return=m,h):(h=i(h,g),h.return=m,h)}function c(m,h,g){if(typeof h=="string"&&h!==""||typeof h=="number")return h=ps(""+h,m.mode,g),h.return=m,h;if(typeof h=="object"&&h!==null){switch(h.$$typeof){case Zi:return g=Ra(h.type,h.key,h.props,null,m.mode,g),g.ref=Vr(m,null,h),g.return=m,g;case tr:return h=fs(h,m.mode,g),h.return=m,h;case an:var C=h._init;return c(m,C(h._payload),g)}if(Gr(h)||Or(h))return h=Wn(h,m.mode,g,null),h.return=m,h;ua(m,h)}return null}function p(m,h,g,C){var E=h!==null?h.key:null;if(typeof g=="string"&&g!==""||typeof g=="number")return E!==null?null:s(m,h,""+g,C);if(typeof g=="object"&&g!==null){switch(g.$$typeof){case Zi:return g.key===E?l(m,h,g,C):null;case tr:return g.key===E?u(m,h,g,C):null;case an:return E=g._init,p(m,h,E(g._payload),C)}if(Gr(g)||Or(g))return E!==null?null:d(m,h,g,C,null);ua(m,g)}return null}function f(m,h,g,C,E){if(typeof C=="string"&&C!==""||typeof C=="number")return m=m.get(g)||null,s(h,m,""+C,E);if(typeof C=="object"&&C!==null){switch(C.$$typeof){case Zi:return m=m.get(C.key===null?g:C.key)||null,l(h,m,C,E);case tr:return m=m.get(C.key===null?g:C.key)||null,u(h,m,C,E);case an:var b=C._init;return f(m,h,g,b(C._payload),E)}if(Gr(C)||Or(C))return m=m.get(g)||null,d(h,m,C,E,null);ua(h,C)}return null}function y(m,h,g,C){for(var E=null,b=null,P=h,T=h=0,D=null;P!==null&&T<g.length;T++){P.index>T?(D=P,P=null):D=P.sibling;var k=p(m,P,g[T],C);if(k===null){P===null&&(P=D);break}e&&P&&k.alternate===null&&t(m,P),h=a(k,h,T),b===null?E=k:b.sibling=k,b=k,P=D}if(T===g.length)return n(m,P),he&&jn(m,T),E;if(P===null){for(;T<g.length;T++)P=c(m,g[T],C),P!==null&&(h=a(P,h,T),b===null?E=P:b.sibling=P,b=P);return he&&jn(m,T),E}for(P=r(m,P);T<g.length;T++)D=f(P,m,T,g[T],C),D!==null&&(e&&D.alternate!==null&&P.delete(D.key===null?T:D.key),h=a(D,h,T),b===null?E=D:b.sibling=D,b=D);return e&&P.forEach(function(I){return t(m,I)}),he&&jn(m,T),E}function _(m,h,g,C){var E=Or(g);if(typeof E!="function")throw Error(A(150));if(g=E.call(g),g==null)throw Error(A(151));for(var b=E=null,P=h,T=h=0,D=null,k=g.next();P!==null&&!k.done;T++,k=g.next()){P.index>T?(D=P,P=null):D=P.sibling;var I=p(m,P,k.value,C);if(I===null){P===null&&(P=D);break}e&&P&&I.alternate===null&&t(m,P),h=a(I,h,T),b===null?E=I:b.sibling=I,b=I,P=D}if(k.done)return n(m,P),he&&jn(m,T),E;if(P===null){for(;!k.done;T++,k=g.next())k=c(m,k.value,C),k!==null&&(h=a(k,h,T),b===null?E=k:b.sibling=k,b=k);return he&&jn(m,T),E}for(P=r(m,P);!k.done;T++,k=g.next())k=f(P,m,T,k.value,C),k!==null&&(e&&k.alternate!==null&&P.delete(k.key===null?T:k.key),h=a(k,h,T),b===null?E=k:b.sibling=k,b=k);return e&&P.forEach(function(W){return t(m,W)}),he&&jn(m,T),E}function S(m,h,g,C){if(typeof g=="object"&&g!==null&&g.type===nr&&g.key===null&&(g=g.props.children),typeof g=="object"&&g!==null){switch(g.$$typeof){case Zi:e:{for(var E=g.key,b=h;b!==null;){if(b.key===E){if(E=g.type,E===nr){if(b.tag===7){n(m,b.sibling),h=i(b,g.props.children),h.return=m,m=h;break e}}else if(b.elementType===E||typeof E=="object"&&E!==null&&E.$$typeof===an&&Hc(E)===b.type){n(m,b.sibling),h=i(b,g.props),h.ref=Vr(m,b,g),h.return=m,m=h;break e}n(m,b);break}else t(m,b);b=b.sibling}g.type===nr?(h=Wn(g.props.children,m.mode,C,g.key),h.return=m,m=h):(C=Ra(g.type,g.key,g.props,null,m.mode,C),C.ref=Vr(m,h,g),C.return=m,m=C)}return o(m);case tr:e:{for(b=g.key;h!==null;){if(h.key===b)if(h.tag===4&&h.stateNode.containerInfo===g.containerInfo&&h.stateNode.implementation===g.implementation){n(m,h.sibling),h=i(h,g.children||[]),h.return=m,m=h;break e}else{n(m,h);break}else t(m,h);h=h.sibling}h=fs(g,m.mode,C),h.return=m,m=h}return o(m);case an:return b=g._init,S(m,h,b(g._payload),C)}if(Gr(g))return y(m,h,g,C);if(Or(g))return _(m,h,g,C);ua(m,g)}return typeof g=="string"&&g!==""||typeof g=="number"?(g=""+g,h!==null&&h.tag===6?(n(m,h.sibling),h=i(h,g),h.return=m,m=h):(n(m,h),h=ps(g,m.mode,C),h.return=m,m=h),o(m)):n(m,h)}return S}var Sr=Vf(!0),Hf=Vf(!1),Ya=Cn(null),Ga=null,cr=null,mu=null;function hu(){mu=cr=Ga=null}function gu(e){var t=Ya.current;me(Ya),e._currentValue=t}function pl(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function xr(e,t){Ga=e,mu=cr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(Ke=!0),e.firstContext=null)}function yt(e){var t=e._currentValue;if(mu!==e)if(e={context:e,memoizedValue:t,next:null},cr===null){if(Ga===null)throw Error(A(308));cr=e,Ga.dependencies={lanes:0,firstContext:e}}else cr=cr.next=e;return t}var Mn=null;function yu(e){Mn===null?Mn=[e]:Mn.push(e)}function $f(e,t,n,r){var i=t.interleaved;return i===null?(n.next=n,yu(t)):(n.next=i.next,i.next=n),t.interleaved=n,Zt(e,r)}function Zt(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var on=!1;function xu(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Qf(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function Kt(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function gn(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,ae&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,Zt(e,n)}return i=r.interleaved,i===null?(t.next=t,yu(r)):(t.next=i.next,i.next=t),r.interleaved=t,Zt(e,n)}function Ea(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,ru(e,n)}}function $c(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function Za(e,t,n,r){var i=e.updateQueue;on=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var l=s,u=l.next;l.next=null,o===null?a=u:o.next=u,o=l;var d=e.alternate;d!==null&&(d=d.updateQueue,s=d.lastBaseUpdate,s!==o&&(s===null?d.firstBaseUpdate=u:s.next=u,d.lastBaseUpdate=l))}if(a!==null){var c=i.baseState;o=0,d=u=l=null,s=a;do{var p=s.lane,f=s.eventTime;if((r&p)===p){d!==null&&(d=d.next={eventTime:f,lane:0,tag:s.tag,payload:s.payload,callback:s.callback,next:null});e:{var y=e,_=s;switch(p=t,f=n,_.tag){case 1:if(y=_.payload,typeof y=="function"){c=y.call(f,c,p);break e}c=y;break e;case 3:y.flags=y.flags&-65537|128;case 0:if(y=_.payload,p=typeof y=="function"?y.call(f,c,p):y,p==null)break e;c=xe({},c,p);break e;case 2:on=!0}}s.callback!==null&&s.lane!==0&&(e.flags|=64,p=i.effects,p===null?i.effects=[s]:p.push(s))}else f={eventTime:f,lane:p,tag:s.tag,payload:s.payload,callback:s.callback,next:null},d===null?(u=d=f,l=c):d=d.next=f,o|=p;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(!0);if(d===null&&(l=c),i.baseState=l,i.firstBaseUpdate=u,i.lastBaseUpdate=d,t=i.shared.interleaved,t!==null){i=t;do o|=i.lane,i=i.next;while(i!==t)}else a===null&&(i.shared.lanes=0);Hn|=o,e.lanes=o,e.memoizedState=c}}function Qc(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],i=r.callback;if(i!==null){if(r.callback=null,r=n,typeof i!="function")throw Error(A(191,i));i.call(r)}}}var Ui={},Ot=Cn(Ui),zi=Cn(Ui),Ei=Cn(Ui);function On(e){if(e===Ui)throw Error(A(174));return e}function _u(e,t){switch(pe(Ei,t),pe(zi,e),pe(Ot,Ui),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:$s(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=$s(t,e)}me(Ot),pe(Ot,t)}function Cr(){me(Ot),me(zi),me(Ei)}function Xf(e){On(Ei.current);var t=On(Ot.current),n=$s(t,e.type);t!==n&&(pe(zi,e),pe(Ot,n))}function vu(e){zi.current===e&&(me(Ot),me(zi))}var ge=Cn(0);function Ja(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var os=[];function wu(){for(var e=0;e<os.length;e++)os[e]._workInProgressVersionPrimary=null;os.length=0}var Pa=en.ReactCurrentDispatcher,ss=en.ReactCurrentBatchConfig,Vn=0,ye=null,Ee=null,Ne=null,eo=!1,si=!1,Pi=0,zx=0;function je(){throw Error(A(321))}function ku(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Lt(e[n],t[n]))return!1;return!0}function bu(e,t,n,r,i,a){if(Vn=a,ye=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,Pa.current=e===null||e.memoizedState===null?Tx:Lx,e=n(r,i),si){a=0;do{if(si=!1,Pi=0,25<=a)throw Error(A(301));a+=1,Ne=Ee=null,t.updateQueue=null,Pa.current=Ax,e=n(r,i)}while(si)}if(Pa.current=to,t=Ee!==null&&Ee.next!==null,Vn=0,Ne=Ee=ye=null,eo=!1,t)throw Error(A(300));return e}function Su(){var e=Pi!==0;return Pi=0,e}function jt(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Ne===null?ye.memoizedState=Ne=e:Ne=Ne.next=e,Ne}function xt(){if(Ee===null){var e=ye.alternate;e=e!==null?e.memoizedState:null}else e=Ee.next;var t=Ne===null?ye.memoizedState:Ne.next;if(t!==null)Ne=t,Ee=e;else{if(e===null)throw Error(A(310));Ee=e,e={memoizedState:Ee.memoizedState,baseState:Ee.baseState,baseQueue:Ee.baseQueue,queue:Ee.queue,next:null},Ne===null?ye.memoizedState=Ne=e:Ne=Ne.next=e}return Ne}function Ni(e,t){return typeof t=="function"?t(e):t}function ls(e){var t=xt(),n=t.queue;if(n===null)throw Error(A(311));n.lastRenderedReducer=e;var r=Ee,i=r.baseQueue,a=n.pending;if(a!==null){if(i!==null){var o=i.next;i.next=a.next,a.next=o}r.baseQueue=i=a,n.pending=null}if(i!==null){a=i.next,r=r.baseState;var s=o=null,l=null,u=a;do{var d=u.lane;if((Vn&d)===d)l!==null&&(l=l.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),r=u.hasEagerState?u.eagerState:e(r,u.action);else{var c={lane:d,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};l===null?(s=l=c,o=r):l=l.next=c,ye.lanes|=d,Hn|=d}u=u.next}while(u!==null&&u!==a);l===null?o=r:l.next=s,Lt(r,t.memoizedState)||(Ke=!0),t.memoizedState=r,t.baseState=o,t.baseQueue=l,n.lastRenderedState=r}if(e=n.interleaved,e!==null){i=e;do a=i.lane,ye.lanes|=a,Hn|=a,i=i.next;while(i!==e)}else i===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function us(e){var t=xt(),n=t.queue;if(n===null)throw Error(A(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,a=t.memoizedState;if(i!==null){n.pending=null;var o=i=i.next;do a=e(a,o.action),o=o.next;while(o!==i);Lt(a,t.memoizedState)||(Ke=!0),t.memoizedState=a,t.baseQueue===null&&(t.baseState=a),n.lastRenderedState=a}return[a,r]}function Kf(){}function Yf(e,t){var n=ye,r=xt(),i=t(),a=!Lt(r.memoizedState,i);if(a&&(r.memoizedState=i,Ke=!0),r=r.queue,Cu(Jf.bind(null,n,r,e),[e]),r.getSnapshot!==t||a||Ne!==null&&Ne.memoizedState.tag&1){if(n.flags|=2048,Ti(9,Zf.bind(null,n,r,i,t),void 0,null),Te===null)throw Error(A(349));Vn&30||Gf(n,t,i)}return i}function Gf(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=ye.updateQueue,t===null?(t={lastEffect:null,stores:null},ye.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function Zf(e,t,n,r){t.value=n,t.getSnapshot=r,em(t)&&tm(e)}function Jf(e,t,n){return n(function(){em(t)&&tm(e)})}function em(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Lt(e,n)}catch{return!0}}function tm(e){var t=Zt(e,1);t!==null&&Nt(t,e,1,-1)}function Xc(e){var t=jt();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:Ni,lastRenderedState:e},t.queue=e,e=e.dispatch=Nx.bind(null,ye,e),[t.memoizedState,e]}function Ti(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=ye.updateQueue,t===null?(t={lastEffect:null,stores:null},ye.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function nm(){return xt().memoizedState}function Na(e,t,n,r){var i=jt();ye.flags|=e,i.memoizedState=Ti(1|t,n,void 0,r===void 0?null:r)}function bo(e,t,n,r){var i=xt();r=r===void 0?null:r;var a=void 0;if(Ee!==null){var o=Ee.memoizedState;if(a=o.destroy,r!==null&&ku(r,o.deps)){i.memoizedState=Ti(t,n,a,r);return}}ye.flags|=e,i.memoizedState=Ti(1|t,n,a,r)}function Kc(e,t){return Na(8390656,8,e,t)}function Cu(e,t){return bo(2048,8,e,t)}function rm(e,t){return bo(4,2,e,t)}function im(e,t){return bo(4,4,e,t)}function am(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function om(e,t,n){return n=n!=null?n.concat([e]):null,bo(4,4,am.bind(null,t,e),n)}function zu(){}function sm(e,t){var n=xt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&ku(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function lm(e,t){var n=xt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&ku(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function um(e,t,n){return Vn&21?(Lt(n,t)||(n=mf(),ye.lanes|=n,Hn|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,Ke=!0),e.memoizedState=n)}function Ex(e,t){var n=se;se=n!==0&&4>n?n:4,e(!0);var r=ss.transition;ss.transition={};try{e(!1),t()}finally{se=n,ss.transition=r}}function cm(){return xt().memoizedState}function Px(e,t,n){var r=xn(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},dm(e))pm(t,n);else if(n=$f(e,t,n,r),n!==null){var i=Ve();Nt(n,e,r,i),fm(n,t,r)}}function Nx(e,t,n){var r=xn(e),i={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(dm(e))pm(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,Lt(s,o)){var l=t.interleaved;l===null?(i.next=i,yu(t)):(i.next=l.next,l.next=i),t.interleaved=i;return}}catch{}finally{}n=$f(e,t,i,r),n!==null&&(i=Ve(),Nt(n,e,r,i),fm(n,t,r))}}function dm(e){var t=e.alternate;return e===ye||t!==null&&t===ye}function pm(e,t){si=eo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function fm(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,ru(e,n)}}var to={readContext:yt,useCallback:je,useContext:je,useEffect:je,useImperativeHandle:je,useInsertionEffect:je,useLayoutEffect:je,useMemo:je,useReducer:je,useRef:je,useState:je,useDebugValue:je,useDeferredValue:je,useTransition:je,useMutableSource:je,useSyncExternalStore:je,useId:je,unstable_isNewReconciler:!1},Tx={readContext:yt,useCallback:function(e,t){return jt().memoizedState=[e,t===void 0?null:t],e},useContext:yt,useEffect:Kc,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,Na(4194308,4,am.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Na(4194308,4,e,t)},useInsertionEffect:function(e,t){return Na(4,2,e,t)},useMemo:function(e,t){var n=jt();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=jt();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=Px.bind(null,ye,e),[r.memoizedState,e]},useRef:function(e){var t=jt();return e={current:e},t.memoizedState=e},useState:Xc,useDebugValue:zu,useDeferredValue:function(e){return jt().memoizedState=e},useTransition:function(){var e=Xc(!1),t=e[0];return e=Ex.bind(null,e[1]),jt().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=ye,i=jt();if(he){if(n===void 0)throw Error(A(407));n=n()}else{if(n=t(),Te===null)throw Error(A(349));Vn&30||Gf(r,t,n)}i.memoizedState=n;var a={value:n,getSnapshot:t};return i.queue=a,Kc(Jf.bind(null,r,a,e),[e]),r.flags|=2048,Ti(9,Zf.bind(null,r,a,n,t),void 0,null),n},useId:function(){var e=jt(),t=Te.identifierPrefix;if(he){var n=Xt,r=Qt;n=(r&~(1<<32-Pt(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=Pi++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=zx++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},Lx={readContext:yt,useCallback:sm,useContext:yt,useEffect:Cu,useImperativeHandle:om,useInsertionEffect:rm,useLayoutEffect:im,useMemo:lm,useReducer:ls,useRef:nm,useState:function(){return ls(Ni)},useDebugValue:zu,useDeferredValue:function(e){var t=xt();return um(t,Ee.memoizedState,e)},useTransition:function(){var e=ls(Ni)[0],t=xt().memoizedState;return[e,t]},useMutableSource:Kf,useSyncExternalStore:Yf,useId:cm,unstable_isNewReconciler:!1},Ax={readContext:yt,useCallback:sm,useContext:yt,useEffect:Cu,useImperativeHandle:om,useInsertionEffect:rm,useLayoutEffect:im,useMemo:lm,useReducer:us,useRef:nm,useState:function(){return us(Ni)},useDebugValue:zu,useDeferredValue:function(e){var t=xt();return Ee===null?t.memoizedState=e:um(t,Ee.memoizedState,e)},useTransition:function(){var e=us(Ni)[0],t=xt().memoizedState;return[e,t]},useMutableSource:Kf,useSyncExternalStore:Yf,useId:cm,unstable_isNewReconciler:!1};function Ct(e,t){if(e&&e.defaultProps){t=xe({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function fl(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:xe({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var So={isMounted:function(e){return(e=e._reactInternals)?Kn(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=Ve(),i=xn(e),a=Kt(r,i);a.payload=t,n!=null&&(a.callback=n),t=gn(e,a,i),t!==null&&(Nt(t,e,i,r),Ea(t,e,i))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=Ve(),i=xn(e),a=Kt(r,i);a.tag=1,a.payload=t,n!=null&&(a.callback=n),t=gn(e,a,i),t!==null&&(Nt(t,e,i,r),Ea(t,e,i))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=Ve(),r=xn(e),i=Kt(n,r);i.tag=2,t!=null&&(i.callback=t),t=gn(e,i,r),t!==null&&(Nt(t,e,r,n),Ea(t,e,r))}};function Yc(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!ki(n,r)||!ki(i,a):!0}function mm(e,t,n){var r=!1,i=bn,a=t.contextType;return typeof a=="object"&&a!==null?a=yt(a):(i=Ge(t)?Bn:Oe.current,r=t.contextTypes,a=(r=r!=null)?kr(e,i):bn),t=new t(n,a),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=So,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=i,e.__reactInternalMemoizedMaskedChildContext=a),t}function Gc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&So.enqueueReplaceState(t,t.state,null)}function ml(e,t,n,r){var i=e.stateNode;i.props=n,i.state=e.memoizedState,i.refs={},xu(e);var a=t.contextType;typeof a=="object"&&a!==null?i.context=yt(a):(a=Ge(t)?Bn:Oe.current,i.context=kr(e,a)),i.state=e.memoizedState,a=t.getDerivedStateFromProps,typeof a=="function"&&(fl(e,t,a,n),i.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof i.getSnapshotBeforeUpdate=="function"||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(t=i.state,typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount(),t!==i.state&&So.enqueueReplaceState(i,i.state,null),Za(e,n,i,r),i.state=e.memoizedState),typeof i.componentDidMount=="function"&&(e.flags|=4194308)}function zr(e,t){try{var n="",r=t;do n+=oy(r),r=r.return;while(r);var i=n}catch(a){i=`
Error generating stack: `+a.message+`
`+a.stack}return{value:e,source:t,stack:i,digest:null}}function cs(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function hl(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var Rx=typeof WeakMap=="function"?WeakMap:Map;function hm(e,t,n){n=Kt(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){ro||(ro=!0,Cl=r),hl(e,t)},n}function gm(e,t,n){n=Kt(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var i=t.value;n.payload=function(){return r(i)},n.callback=function(){hl(e,t)}}var a=e.stateNode;return a!==null&&typeof a.componentDidCatch=="function"&&(n.callback=function(){hl(e,t),typeof r!="function"&&(yn===null?yn=new Set([this]):yn.add(this));var o=t.stack;this.componentDidCatch(t.value,{componentStack:o!==null?o:""})}),n}function Zc(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new Rx;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(i.add(n),e=Qx.bind(null,e,t,n),t.then(e,e))}function Jc(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function ed(e,t,n,r,i){return e.mode&1?(e.flags|=65536,e.lanes=i,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=Kt(-1,1),t.tag=2,gn(n,t,1))),n.lanes|=1),e)}var jx=en.ReactCurrentOwner,Ke=!1;function Be(e,t,n,r){t.child=e===null?Hf(t,null,n,r):Sr(t,e.child,n,r)}function td(e,t,n,r,i){n=n.render;var a=t.ref;return xr(t,i),r=bu(e,t,n,r,a,i),n=Su(),e!==null&&!Ke?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,Jt(e,t,i)):(he&&n&&du(t),t.flags|=1,Be(e,t,r,i),t.child)}function nd(e,t,n,r,i){if(e===null){var a=n.type;return typeof a=="function"&&!ju(a)&&a.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=a,ym(e,t,a,r,i)):(e=Ra(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!(e.lanes&i)){var o=a.memoizedProps;if(n=n.compare,n=n!==null?n:ki,n(o,r)&&e.ref===t.ref)return Jt(e,t,i)}return t.flags|=1,e=_n(a,r),e.ref=t.ref,e.return=t,t.child=e}function ym(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(ki(a,r)&&e.ref===t.ref)if(Ke=!1,t.pendingProps=r=a,(e.lanes&i)!==0)e.flags&131072&&(Ke=!0);else return t.lanes=e.lanes,Jt(e,t,i)}return gl(e,t,n,r,i)}function xm(e,t,n){var r=t.pendingProps,i=r.children,a=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},pe(pr,nt),nt|=n;else{if(!(n&1073741824))return e=a!==null?a.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,pe(pr,nt),nt|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=a!==null?a.baseLanes:n,pe(pr,nt),nt|=r}else a!==null?(r=a.baseLanes|n,t.memoizedState=null):r=n,pe(pr,nt),nt|=r;return Be(e,t,i,n),t.child}function _m(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function gl(e,t,n,r,i){var a=Ge(n)?Bn:Oe.current;return a=kr(t,a),xr(t,i),n=bu(e,t,n,r,a,i),r=Su(),e!==null&&!Ke?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,Jt(e,t,i)):(he&&r&&du(t),t.flags|=1,Be(e,t,n,i),t.child)}function rd(e,t,n,r,i){if(Ge(n)){var a=!0;Qa(t)}else a=!1;if(xr(t,i),t.stateNode===null)Ta(e,t),mm(t,n,r),ml(t,n,r,i),r=!0;else if(e===null){var o=t.stateNode,s=t.memoizedProps;o.props=s;var l=o.context,u=n.contextType;typeof u=="object"&&u!==null?u=yt(u):(u=Ge(n)?Bn:Oe.current,u=kr(t,u));var d=n.getDerivedStateFromProps,c=typeof d=="function"||typeof o.getSnapshotBeforeUpdate=="function";c||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==r||l!==u)&&Gc(t,o,r,u),on=!1;var p=t.memoizedState;o.state=p,Za(t,r,o,i),l=t.memoizedState,s!==r||p!==l||Ye.current||on?(typeof d=="function"&&(fl(t,n,d,r),l=t.memoizedState),(s=on||Yc(t,n,s,r,p,l,u))?(c||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(t.flags|=4194308)):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),o.props=r,o.state=l,o.context=u,r=s):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{o=t.stateNode,Qf(e,t),s=t.memoizedProps,u=t.type===t.elementType?s:Ct(t.type,s),o.props=u,c=t.pendingProps,p=o.context,l=n.contextType,typeof l=="object"&&l!==null?l=yt(l):(l=Ge(n)?Bn:Oe.current,l=kr(t,l));var f=n.getDerivedStateFromProps;(d=typeof f=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==c||p!==l)&&Gc(t,o,r,l),on=!1,p=t.memoizedState,o.state=p,Za(t,r,o,i);var y=t.memoizedState;s!==c||p!==y||Ye.current||on?(typeof f=="function"&&(fl(t,n,f,r),y=t.memoizedState),(u=on||Yc(t,n,u,r,p,y,l)||!1)?(d||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(r,y,l),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(r,y,l)),typeof o.componentDidUpdate=="function"&&(t.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=y),o.props=r,o.state=y,o.context=l,r=u):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=1024),r=!1)}return yl(e,t,n,r,a,i)}function yl(e,t,n,r,i,a){_m(e,t);var o=(t.flags&128)!==0;if(!r&&!o)return i&&Bc(t,n,!1),Jt(e,t,a);r=t.stateNode,jx.current=t;var s=o&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&o?(t.child=Sr(t,e.child,null,a),t.child=Sr(t,null,s,a)):Be(e,t,s,a),t.memoizedState=r.state,i&&Bc(t,n,!0),t.child}function vm(e){var t=e.stateNode;t.pendingContext?Wc(e,t.pendingContext,t.pendingContext!==t.context):t.context&&Wc(e,t.context,!1),_u(e,t.containerInfo)}function id(e,t,n,r,i){return br(),fu(i),t.flags|=256,Be(e,t,n,r),t.child}var xl={dehydrated:null,treeContext:null,retryLane:0};function _l(e){return{baseLanes:e,cachePool:null,transitions:null}}function wm(e,t,n){var r=t.pendingProps,i=ge.current,a=!1,o=(t.flags&128)!==0,s;if((s=o)||(s=e!==null&&e.memoizedState===null?!1:(i&2)!==0),s?(a=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(i|=1),pe(ge,i&1),e===null)return dl(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(o=r.children,e=r.fallback,a?(r=t.mode,a=t.child,o={mode:"hidden",children:o},!(r&1)&&a!==null?(a.childLanes=0,a.pendingProps=o):a=Eo(o,r,0,null),e=Wn(e,r,n,null),a.return=t,e.return=t,a.sibling=e,t.child=a,t.child.memoizedState=_l(n),t.memoizedState=xl,e):Eu(t,o));if(i=e.memoizedState,i!==null&&(s=i.dehydrated,s!==null))return Ix(e,t,o,r,s,i,n);if(a){a=r.fallback,o=t.mode,i=e.child,s=i.sibling;var l={mode:"hidden",children:r.children};return!(o&1)&&t.child!==i?(r=t.child,r.childLanes=0,r.pendingProps=l,t.deletions=null):(r=_n(i,l),r.subtreeFlags=i.subtreeFlags&14680064),s!==null?a=_n(s,a):(a=Wn(a,o,n,null),a.flags|=2),a.return=t,r.return=t,r.sibling=a,t.child=r,r=a,a=t.child,o=e.child.memoizedState,o=o===null?_l(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},a.memoizedState=o,a.childLanes=e.childLanes&~n,t.memoizedState=xl,r}return a=e.child,e=a.sibling,r=_n(a,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function Eu(e,t){return t=Eo({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function ca(e,t,n,r){return r!==null&&fu(r),Sr(t,e.child,null,n),e=Eu(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function Ix(e,t,n,r,i,a,o){if(n)return t.flags&256?(t.flags&=-257,r=cs(Error(A(422))),ca(e,t,o,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(a=r.fallback,i=t.mode,r=Eo({mode:"visible",children:r.children},i,0,null),a=Wn(a,i,o,null),a.flags|=2,r.return=t,a.return=t,r.sibling=a,t.child=r,t.mode&1&&Sr(t,e.child,null,o),t.child.memoizedState=_l(o),t.memoizedState=xl,a);if(!(t.mode&1))return ca(e,t,o,null);if(i.data==="$!"){if(r=i.nextSibling&&i.nextSibling.dataset,r)var s=r.dgst;return r=s,a=Error(A(419)),r=cs(a,r,void 0),ca(e,t,o,r)}if(s=(o&e.childLanes)!==0,Ke||s){if(r=Te,r!==null){switch(o&-o){case 4:i=2;break;case 16:i=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:i=32;break;case 536870912:i=268435456;break;default:i=0}i=i&(r.suspendedLanes|o)?0:i,i!==0&&i!==a.retryLane&&(a.retryLane=i,Zt(e,i),Nt(r,e,i,-1))}return Ru(),r=cs(Error(A(421))),ca(e,t,o,r)}return i.data==="$?"?(t.flags|=128,t.child=e.child,t=Xx.bind(null,e),i._reactRetry=t,null):(e=a.treeContext,it=hn(i.nextSibling),at=t,he=!0,Et=null,e!==null&&(pt[ft++]=Qt,pt[ft++]=Xt,pt[ft++]=Un,Qt=e.id,Xt=e.overflow,Un=t),t=Eu(t,r.children),t.flags|=4096,t)}function ad(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),pl(e.return,t,n)}function ds(e,t,n,r,i){var a=e.memoizedState;a===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i}:(a.isBackwards=t,a.rendering=null,a.renderingStartTime=0,a.last=r,a.tail=n,a.tailMode=i)}function km(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;if(Be(e,t,r.children,n),r=ge.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&ad(e,n,t);else if(e.tag===19)ad(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(pe(ge,r),!(t.mode&1))t.memoizedState=null;else switch(i){case"forwards":for(n=t.child,i=null;n!==null;)e=n.alternate,e!==null&&Ja(e)===null&&(i=n),n=n.sibling;n=i,n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),ds(t,!1,i,n,a);break;case"backwards":for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Ja(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}ds(t,!0,n,null,a);break;case"together":ds(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Ta(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function Jt(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Hn|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(A(153));if(t.child!==null){for(e=t.child,n=_n(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=_n(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function Dx(e,t,n){switch(t.tag){case 3:vm(t),br();break;case 5:Xf(t);break;case 1:Ge(t.type)&&Qa(t);break;case 4:_u(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,i=t.memoizedProps.value;pe(Ya,r._currentValue),r._currentValue=i;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(pe(ge,ge.current&1),t.flags|=128,null):n&t.child.childLanes?wm(e,t,n):(pe(ge,ge.current&1),e=Jt(e,t,n),e!==null?e.sibling:null);pe(ge,ge.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return km(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),pe(ge,ge.current),r)break;return null;case 22:case 23:return t.lanes=0,xm(e,t,n)}return Jt(e,t,n)}var bm,vl,Sm,Cm;bm=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};vl=function(){};Sm=function(e,t,n,r){var i=e.memoizedProps;if(i!==r){e=t.stateNode,On(Ot.current);var a=null;switch(n){case"input":i=Bs(e,i),r=Bs(e,r),a=[];break;case"select":i=xe({},i,{value:void 0}),r=xe({},r,{value:void 0}),a=[];break;case"textarea":i=Hs(e,i),r=Hs(e,r),a=[];break;default:typeof i.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=Ha)}Qs(n,r);var o;n=null;for(u in i)if(!r.hasOwnProperty(u)&&i.hasOwnProperty(u)&&i[u]!=null)if(u==="style"){var s=i[u];for(o in s)s.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(hi.hasOwnProperty(u)?a||(a=[]):(a=a||[]).push(u,null));for(u in r){var l=r[u];if(s=i!=null?i[u]:void 0,r.hasOwnProperty(u)&&l!==s&&(l!=null||s!=null))if(u==="style")if(s){for(o in s)!s.hasOwnProperty(o)||l&&l.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in l)l.hasOwnProperty(o)&&s[o]!==l[o]&&(n||(n={}),n[o]=l[o])}else n||(a||(a=[]),a.push(u,n)),n=l;else u==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,s=s?s.__html:void 0,l!=null&&s!==l&&(a=a||[]).push(u,l)):u==="children"?typeof l!="string"&&typeof l!="number"||(a=a||[]).push(u,""+l):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(hi.hasOwnProperty(u)?(l!=null&&u==="onScroll"&&fe("scroll",e),a||s===l||(a=[])):(a=a||[]).push(u,l))}n&&(a=a||[]).push("style",n);var u=a;(t.updateQueue=u)&&(t.flags|=4)}};Cm=function(e,t,n,r){n!==r&&(t.flags|=4)};function Hr(e,t){if(!he)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function Ie(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&14680064,r|=i.flags&14680064,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function Fx(e,t,n){var r=t.pendingProps;switch(pu(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Ie(t),null;case 1:return Ge(t.type)&&$a(),Ie(t),null;case 3:return r=t.stateNode,Cr(),me(Ye),me(Oe),wu(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(la(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,Et!==null&&(Pl(Et),Et=null))),vl(e,t),Ie(t),null;case 5:vu(t);var i=On(Ei.current);if(n=t.type,e!==null&&t.stateNode!=null)Sm(e,t,n,r,i),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(A(166));return Ie(t),null}if(e=On(Ot.current),la(t)){r=t.stateNode,n=t.type;var a=t.memoizedProps;switch(r[Dt]=t,r[Ci]=a,e=(t.mode&1)!==0,n){case"dialog":fe("cancel",r),fe("close",r);break;case"iframe":case"object":case"embed":fe("load",r);break;case"video":case"audio":for(i=0;i<Jr.length;i++)fe(Jr[i],r);break;case"source":fe("error",r);break;case"img":case"image":case"link":fe("error",r),fe("load",r);break;case"details":fe("toggle",r);break;case"input":mc(r,a),fe("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!a.multiple},fe("invalid",r);break;case"textarea":gc(r,a),fe("invalid",r)}Qs(n,a),i=null;for(var o in a)if(a.hasOwnProperty(o)){var s=a[o];o==="children"?typeof s=="string"?r.textContent!==s&&(a.suppressHydrationWarning!==!0&&sa(r.textContent,s,e),i=["children",s]):typeof s=="number"&&r.textContent!==""+s&&(a.suppressHydrationWarning!==!0&&sa(r.textContent,s,e),i=["children",""+s]):hi.hasOwnProperty(o)&&s!=null&&o==="onScroll"&&fe("scroll",r)}switch(n){case"input":Ji(r),hc(r,a,!0);break;case"textarea":Ji(r),yc(r);break;case"select":case"option":break;default:typeof a.onClick=="function"&&(r.onclick=Ha)}r=i,t.updateQueue=r,r!==null&&(t.flags|=4)}else{o=i.nodeType===9?i:i.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Zp(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=o.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=o.createElement(n,{is:r.is}):(e=o.createElement(n),n==="select"&&(o=e,r.multiple?o.multiple=!0:r.size&&(o.size=r.size))):e=o.createElementNS(e,n),e[Dt]=t,e[Ci]=r,bm(e,t,!1,!1),t.stateNode=e;e:{switch(o=Xs(n,r),n){case"dialog":fe("cancel",e),fe("close",e),i=r;break;case"iframe":case"object":case"embed":fe("load",e),i=r;break;case"video":case"audio":for(i=0;i<Jr.length;i++)fe(Jr[i],e);i=r;break;case"source":fe("error",e),i=r;break;case"img":case"image":case"link":fe("error",e),fe("load",e),i=r;break;case"details":fe("toggle",e),i=r;break;case"input":mc(e,r),i=Bs(e,r),fe("invalid",e);break;case"option":i=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},i=xe({},r,{value:void 0}),fe("invalid",e);break;case"textarea":gc(e,r),i=Hs(e,r),fe("invalid",e);break;default:i=r}Qs(n,i),s=i;for(a in s)if(s.hasOwnProperty(a)){var l=s[a];a==="style"?tf(e,l):a==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,l!=null&&Jp(e,l)):a==="children"?typeof l=="string"?(n!=="textarea"||l!=="")&&gi(e,l):typeof l=="number"&&gi(e,""+l):a!=="suppressContentEditableWarning"&&a!=="suppressHydrationWarning"&&a!=="autoFocus"&&(hi.hasOwnProperty(a)?l!=null&&a==="onScroll"&&fe("scroll",e):l!=null&&Gl(e,a,l,o))}switch(n){case"input":Ji(e),hc(e,r,!1);break;case"textarea":Ji(e),yc(e);break;case"option":r.value!=null&&e.setAttribute("value",""+kn(r.value));break;case"select":e.multiple=!!r.multiple,a=r.value,a!=null?mr(e,!!r.multiple,a,!1):r.defaultValue!=null&&mr(e,!!r.multiple,r.defaultValue,!0);break;default:typeof i.onClick=="function"&&(e.onclick=Ha)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return Ie(t),null;case 6:if(e&&t.stateNode!=null)Cm(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(A(166));if(n=On(Ei.current),On(Ot.current),la(t)){if(r=t.stateNode,n=t.memoizedProps,r[Dt]=t,(a=r.nodeValue!==n)&&(e=at,e!==null))switch(e.tag){case 3:sa(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&sa(r.nodeValue,n,(e.mode&1)!==0)}a&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[Dt]=t,t.stateNode=r}return Ie(t),null;case 13:if(me(ge),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(he&&it!==null&&t.mode&1&&!(t.flags&128))Uf(),br(),t.flags|=98560,a=!1;else if(a=la(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(A(318));if(a=t.memoizedState,a=a!==null?a.dehydrated:null,!a)throw Error(A(317));a[Dt]=t}else br(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;Ie(t),a=!1}else Et!==null&&(Pl(Et),Et=null),a=!0;if(!a)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||ge.current&1?Pe===0&&(Pe=3):Ru())),t.updateQueue!==null&&(t.flags|=4),Ie(t),null);case 4:return Cr(),vl(e,t),e===null&&bi(t.stateNode.containerInfo),Ie(t),null;case 10:return gu(t.type._context),Ie(t),null;case 17:return Ge(t.type)&&$a(),Ie(t),null;case 19:if(me(ge),a=t.memoizedState,a===null)return Ie(t),null;if(r=(t.flags&128)!==0,o=a.rendering,o===null)if(r)Hr(a,!1);else{if(Pe!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=Ja(e),o!==null){for(t.flags|=128,Hr(a,!1),r=o.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)a=n,e=r,a.flags&=14680066,o=a.alternate,o===null?(a.childLanes=0,a.lanes=e,a.child=null,a.subtreeFlags=0,a.memoizedProps=null,a.memoizedState=null,a.updateQueue=null,a.dependencies=null,a.stateNode=null):(a.childLanes=o.childLanes,a.lanes=o.lanes,a.child=o.child,a.subtreeFlags=0,a.deletions=null,a.memoizedProps=o.memoizedProps,a.memoizedState=o.memoizedState,a.updateQueue=o.updateQueue,a.type=o.type,e=o.dependencies,a.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return pe(ge,ge.current&1|2),t.child}e=e.sibling}a.tail!==null&&be()>Er&&(t.flags|=128,r=!0,Hr(a,!1),t.lanes=4194304)}else{if(!r)if(e=Ja(o),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),Hr(a,!0),a.tail===null&&a.tailMode==="hidden"&&!o.alternate&&!he)return Ie(t),null}else 2*be()-a.renderingStartTime>Er&&n!==1073741824&&(t.flags|=128,r=!0,Hr(a,!1),t.lanes=4194304);a.isBackwards?(o.sibling=t.child,t.child=o):(n=a.last,n!==null?n.sibling=o:t.child=o,a.last=o)}return a.tail!==null?(t=a.tail,a.rendering=t,a.tail=t.sibling,a.renderingStartTime=be(),t.sibling=null,n=ge.current,pe(ge,r?n&1|2:n&1),t):(Ie(t),null);case 22:case 23:return Au(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?nt&1073741824&&(Ie(t),t.subtreeFlags&6&&(t.flags|=8192)):Ie(t),null;case 24:return null;case 25:return null}throw Error(A(156,t.tag))}function Mx(e,t){switch(pu(t),t.tag){case 1:return Ge(t.type)&&$a(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return Cr(),me(Ye),me(Oe),wu(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return vu(t),null;case 13:if(me(ge),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(A(340));br()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return me(ge),null;case 4:return Cr(),null;case 10:return gu(t.type._context),null;case 22:case 23:return Au(),null;case 24:return null;default:return null}}var da=!1,Fe=!1,Ox=typeof WeakSet=="function"?WeakSet:Set,U=null;function dr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){ve(e,t,r)}else n.current=null}function wl(e,t,n){try{n()}catch(r){ve(e,t,r)}}var od=!1;function qx(e,t){if(il=Ba,e=Tf(),cu(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var i=r.anchorOffset,a=r.focusNode;r=r.focusOffset;try{n.nodeType,a.nodeType}catch{n=null;break e}var o=0,s=-1,l=-1,u=0,d=0,c=e,p=null;t:for(;;){for(var f;c!==n||i!==0&&c.nodeType!==3||(s=o+i),c!==a||r!==0&&c.nodeType!==3||(l=o+r),c.nodeType===3&&(o+=c.nodeValue.length),(f=c.firstChild)!==null;)p=c,c=f;for(;;){if(c===e)break t;if(p===n&&++u===i&&(s=o),p===a&&++d===r&&(l=o),(f=c.nextSibling)!==null)break;c=p,p=c.parentNode}c=f}n=s===-1||l===-1?null:{start:s,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(al={focusedElem:e,selectionRange:n},Ba=!1,U=t;U!==null;)if(t=U,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,U=e;else for(;U!==null;){t=U;try{var y=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(y!==null){var _=y.memoizedProps,S=y.memoizedState,m=t.stateNode,h=m.getSnapshotBeforeUpdate(t.elementType===t.type?_:Ct(t.type,_),S);m.__reactInternalSnapshotBeforeUpdate=h}break;case 3:var g=t.stateNode.containerInfo;g.nodeType===1?g.textContent="":g.nodeType===9&&g.documentElement&&g.removeChild(g.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(A(163))}}catch(C){ve(t,t.return,C)}if(e=t.sibling,e!==null){e.return=t.return,U=e;break}U=t.return}return y=od,od=!1,y}function li(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var i=r=r.next;do{if((i.tag&e)===e){var a=i.destroy;i.destroy=void 0,a!==void 0&&wl(t,n,a)}i=i.next}while(i!==r)}}function Co(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function kl(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function zm(e){var t=e.alternate;t!==null&&(e.alternate=null,zm(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[Dt],delete t[Ci],delete t[ll],delete t[kx],delete t[bx])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function Em(e){return e.tag===5||e.tag===3||e.tag===4}function sd(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||Em(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function bl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=Ha));else if(r!==4&&(e=e.child,e!==null))for(bl(e,t,n),e=e.sibling;e!==null;)bl(e,t,n),e=e.sibling}function Sl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(Sl(e,t,n),e=e.sibling;e!==null;)Sl(e,t,n),e=e.sibling}var Le=null,zt=!1;function nn(e,t,n){for(n=n.child;n!==null;)Pm(e,t,n),n=n.sibling}function Pm(e,t,n){if(Mt&&typeof Mt.onCommitFiberUnmount=="function")try{Mt.onCommitFiberUnmount(yo,n)}catch{}switch(n.tag){case 5:Fe||dr(n,t);case 6:var r=Le,i=zt;Le=null,nn(e,t,n),Le=r,zt=i,Le!==null&&(zt?(e=Le,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Le.removeChild(n.stateNode));break;case 18:Le!==null&&(zt?(e=Le,n=n.stateNode,e.nodeType===8?is(e.parentNode,n):e.nodeType===1&&is(e,n),vi(e)):is(Le,n.stateNode));break;case 4:r=Le,i=zt,Le=n.stateNode.containerInfo,zt=!0,nn(e,t,n),Le=r,zt=i;break;case 0:case 11:case 14:case 15:if(!Fe&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){i=r=r.next;do{var a=i,o=a.destroy;a=a.tag,o!==void 0&&(a&2||a&4)&&wl(n,t,o),i=i.next}while(i!==r)}nn(e,t,n);break;case 1:if(!Fe&&(dr(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(s){ve(n,t,s)}nn(e,t,n);break;case 21:nn(e,t,n);break;case 22:n.mode&1?(Fe=(r=Fe)||n.memoizedState!==null,nn(e,t,n),Fe=r):nn(e,t,n);break;default:nn(e,t,n)}}function ld(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new Ox),t.forEach(function(r){var i=Kx.bind(null,e,r);n.has(r)||(n.add(r),r.then(i,i))})}}function vt(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];try{var a=e,o=t,s=o;e:for(;s!==null;){switch(s.tag){case 5:Le=s.stateNode,zt=!1;break e;case 3:Le=s.stateNode.containerInfo,zt=!0;break e;case 4:Le=s.stateNode.containerInfo,zt=!0;break e}s=s.return}if(Le===null)throw Error(A(160));Pm(a,o,i),Le=null,zt=!1;var l=i.alternate;l!==null&&(l.return=null),i.return=null}catch(u){ve(i,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)Nm(t,e),t=t.sibling}function Nm(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(vt(t,e),Rt(e),r&4){try{li(3,e,e.return),Co(3,e)}catch(_){ve(e,e.return,_)}try{li(5,e,e.return)}catch(_){ve(e,e.return,_)}}break;case 1:vt(t,e),Rt(e),r&512&&n!==null&&dr(n,n.return);break;case 5:if(vt(t,e),Rt(e),r&512&&n!==null&&dr(n,n.return),e.flags&32){var i=e.stateNode;try{gi(i,"")}catch(_){ve(e,e.return,_)}}if(r&4&&(i=e.stateNode,i!=null)){var a=e.memoizedProps,o=n!==null?n.memoizedProps:a,s=e.type,l=e.updateQueue;if(e.updateQueue=null,l!==null)try{s==="input"&&a.type==="radio"&&a.name!=null&&Yp(i,a),Xs(s,o);var u=Xs(s,a);for(o=0;o<l.length;o+=2){var d=l[o],c=l[o+1];d==="style"?tf(i,c):d==="dangerouslySetInnerHTML"?Jp(i,c):d==="children"?gi(i,c):Gl(i,d,c,u)}switch(s){case"input":Us(i,a);break;case"textarea":Gp(i,a);break;case"select":var p=i._wrapperState.wasMultiple;i._wrapperState.wasMultiple=!!a.multiple;var f=a.value;f!=null?mr(i,!!a.multiple,f,!1):p!==!!a.multiple&&(a.defaultValue!=null?mr(i,!!a.multiple,a.defaultValue,!0):mr(i,!!a.multiple,a.multiple?[]:"",!1))}i[Ci]=a}catch(_){ve(e,e.return,_)}}break;case 6:if(vt(t,e),Rt(e),r&4){if(e.stateNode===null)throw Error(A(162));i=e.stateNode,a=e.memoizedProps;try{i.nodeValue=a}catch(_){ve(e,e.return,_)}}break;case 3:if(vt(t,e),Rt(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{vi(t.containerInfo)}catch(_){ve(e,e.return,_)}break;case 4:vt(t,e),Rt(e);break;case 13:vt(t,e),Rt(e),i=e.child,i.flags&8192&&(a=i.memoizedState!==null,i.stateNode.isHidden=a,!a||i.alternate!==null&&i.alternate.memoizedState!==null||(Tu=be())),r&4&&ld(e);break;case 22:if(d=n!==null&&n.memoizedState!==null,e.mode&1?(Fe=(u=Fe)||d,vt(t,e),Fe=u):vt(t,e),Rt(e),r&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!d&&e.mode&1)for(U=e,d=e.child;d!==null;){for(c=U=d;U!==null;){switch(p=U,f=p.child,p.tag){case 0:case 11:case 14:case 15:li(4,p,p.return);break;case 1:dr(p,p.return);var y=p.stateNode;if(typeof y.componentWillUnmount=="function"){r=p,n=p.return;try{t=r,y.props=t.memoizedProps,y.state=t.memoizedState,y.componentWillUnmount()}catch(_){ve(r,n,_)}}break;case 5:dr(p,p.return);break;case 22:if(p.memoizedState!==null){cd(c);continue}}f!==null?(f.return=p,U=f):cd(c)}d=d.sibling}e:for(d=null,c=e;;){if(c.tag===5){if(d===null){d=c;try{i=c.stateNode,u?(a=i.style,typeof a.setProperty=="function"?a.setProperty("display","none","important"):a.display="none"):(s=c.stateNode,l=c.memoizedProps.style,o=l!=null&&l.hasOwnProperty("display")?l.display:null,s.style.display=ef("display",o))}catch(_){ve(e,e.return,_)}}}else if(c.tag===6){if(d===null)try{c.stateNode.nodeValue=u?"":c.memoizedProps}catch(_){ve(e,e.return,_)}}else if((c.tag!==22&&c.tag!==23||c.memoizedState===null||c===e)&&c.child!==null){c.child.return=c,c=c.child;continue}if(c===e)break e;for(;c.sibling===null;){if(c.return===null||c.return===e)break e;d===c&&(d=null),c=c.return}d===c&&(d=null),c.sibling.return=c.return,c=c.sibling}}break;case 19:vt(t,e),Rt(e),r&4&&ld(e);break;case 21:break;default:vt(t,e),Rt(e)}}function Rt(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(Em(n)){var r=n;break e}n=n.return}throw Error(A(160))}switch(r.tag){case 5:var i=r.stateNode;r.flags&32&&(gi(i,""),r.flags&=-33);var a=sd(e);Sl(e,a,i);break;case 3:case 4:var o=r.stateNode.containerInfo,s=sd(e);bl(e,s,o);break;default:throw Error(A(161))}}catch(l){ve(e,e.return,l)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Wx(e,t,n){U=e,Tm(e)}function Tm(e,t,n){for(var r=(e.mode&1)!==0;U!==null;){var i=U,a=i.child;if(i.tag===22&&r){var o=i.memoizedState!==null||da;if(!o){var s=i.alternate,l=s!==null&&s.memoizedState!==null||Fe;s=da;var u=Fe;if(da=o,(Fe=l)&&!u)for(U=i;U!==null;)o=U,l=o.child,o.tag===22&&o.memoizedState!==null?dd(i):l!==null?(l.return=o,U=l):dd(i);for(;a!==null;)U=a,Tm(a),a=a.sibling;U=i,da=s,Fe=u}ud(e)}else i.subtreeFlags&8772&&a!==null?(a.return=i,U=a):ud(e)}}function ud(e){for(;U!==null;){var t=U;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Fe||Co(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!Fe)if(n===null)r.componentDidMount();else{var i=t.elementType===t.type?n.memoizedProps:Ct(t.type,n.memoizedProps);r.componentDidUpdate(i,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var a=t.updateQueue;a!==null&&Qc(t,a,r);break;case 3:var o=t.updateQueue;if(o!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}Qc(t,o,n)}break;case 5:var s=t.stateNode;if(n===null&&t.flags&4){n=s;var l=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":l.autoFocus&&n.focus();break;case"img":l.src&&(n.src=l.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var d=u.memoizedState;if(d!==null){var c=d.dehydrated;c!==null&&vi(c)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(A(163))}Fe||t.flags&512&&kl(t)}catch(p){ve(t,t.return,p)}}if(t===e){U=null;break}if(n=t.sibling,n!==null){n.return=t.return,U=n;break}U=t.return}}function cd(e){for(;U!==null;){var t=U;if(t===e){U=null;break}var n=t.sibling;if(n!==null){n.return=t.return,U=n;break}U=t.return}}function dd(e){for(;U!==null;){var t=U;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{Co(4,t)}catch(l){ve(t,n,l)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var i=t.return;try{r.componentDidMount()}catch(l){ve(t,i,l)}}var a=t.return;try{kl(t)}catch(l){ve(t,a,l)}break;case 5:var o=t.return;try{kl(t)}catch(l){ve(t,o,l)}}}catch(l){ve(t,t.return,l)}if(t===e){U=null;break}var s=t.sibling;if(s!==null){s.return=t.return,U=s;break}U=t.return}}var Bx=Math.ceil,no=en.ReactCurrentDispatcher,Pu=en.ReactCurrentOwner,gt=en.ReactCurrentBatchConfig,ae=0,Te=null,Ce=null,Ae=0,nt=0,pr=Cn(0),Pe=0,Li=null,Hn=0,zo=0,Nu=0,ui=null,Xe=null,Tu=0,Er=1/0,Ht=null,ro=!1,Cl=null,yn=null,pa=!1,cn=null,io=0,ci=0,zl=null,La=-1,Aa=0;function Ve(){return ae&6?be():La!==-1?La:La=be()}function xn(e){return e.mode&1?ae&2&&Ae!==0?Ae&-Ae:Cx.transition!==null?(Aa===0&&(Aa=mf()),Aa):(e=se,e!==0||(e=window.event,e=e===void 0?16:wf(e.type)),e):1}function Nt(e,t,n,r){if(50<ci)throw ci=0,zl=null,Error(A(185));qi(e,n,r),(!(ae&2)||e!==Te)&&(e===Te&&(!(ae&2)&&(zo|=n),Pe===4&&ln(e,Ae)),Ze(e,r),n===1&&ae===0&&!(t.mode&1)&&(Er=be()+500,ko&&zn()))}function Ze(e,t){var n=e.callbackNode;Cy(e,t);var r=Wa(e,e===Te?Ae:0);if(r===0)n!==null&&vc(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&vc(n),t===1)e.tag===0?Sx(pd.bind(null,e)):qf(pd.bind(null,e)),vx(function(){!(ae&6)&&zn()}),n=null;else{switch(hf(r)){case 1:n=nu;break;case 4:n=pf;break;case 16:n=qa;break;case 536870912:n=ff;break;default:n=qa}n=Mm(n,Lm.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function Lm(e,t){if(La=-1,Aa=0,ae&6)throw Error(A(327));var n=e.callbackNode;if(_r()&&e.callbackNode!==n)return null;var r=Wa(e,e===Te?Ae:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=ao(e,r);else{t=r;var i=ae;ae|=2;var a=Rm();(Te!==e||Ae!==t)&&(Ht=null,Er=be()+500,qn(e,t));do try{Hx();break}catch(s){Am(e,s)}while(!0);hu(),no.current=a,ae=i,Ce!==null?t=0:(Te=null,Ae=0,t=Pe)}if(t!==0){if(t===2&&(i=Js(e),i!==0&&(r=i,t=El(e,i))),t===1)throw n=Li,qn(e,0),ln(e,r),Ze(e,be()),n;if(t===6)ln(e,r);else{if(i=e.current.alternate,!(r&30)&&!Ux(i)&&(t=ao(e,r),t===2&&(a=Js(e),a!==0&&(r=a,t=El(e,a))),t===1))throw n=Li,qn(e,0),ln(e,r),Ze(e,be()),n;switch(e.finishedWork=i,e.finishedLanes=r,t){case 0:case 1:throw Error(A(345));case 2:In(e,Xe,Ht);break;case 3:if(ln(e,r),(r&130023424)===r&&(t=Tu+500-be(),10<t)){if(Wa(e,0)!==0)break;if(i=e.suspendedLanes,(i&r)!==r){Ve(),e.pingedLanes|=e.suspendedLanes&i;break}e.timeoutHandle=sl(In.bind(null,e,Xe,Ht),t);break}In(e,Xe,Ht);break;case 4:if(ln(e,r),(r&4194240)===r)break;for(t=e.eventTimes,i=-1;0<r;){var o=31-Pt(r);a=1<<o,o=t[o],o>i&&(i=o),r&=~a}if(r=i,r=be()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*Bx(r/1960))-r,10<r){e.timeoutHandle=sl(In.bind(null,e,Xe,Ht),r);break}In(e,Xe,Ht);break;case 5:In(e,Xe,Ht);break;default:throw Error(A(329))}}}return Ze(e,be()),e.callbackNode===n?Lm.bind(null,e):null}function El(e,t){var n=ui;return e.current.memoizedState.isDehydrated&&(qn(e,t).flags|=256),e=ao(e,t),e!==2&&(t=Xe,Xe=n,t!==null&&Pl(t)),e}function Pl(e){Xe===null?Xe=e:Xe.push.apply(Xe,e)}function Ux(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!Lt(a(),i))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function ln(e,t){for(t&=~Nu,t&=~zo,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-Pt(t),r=1<<n;e[n]=-1,t&=~r}}function pd(e){if(ae&6)throw Error(A(327));_r();var t=Wa(e,0);if(!(t&1))return Ze(e,be()),null;var n=ao(e,t);if(e.tag!==0&&n===2){var r=Js(e);r!==0&&(t=r,n=El(e,r))}if(n===1)throw n=Li,qn(e,0),ln(e,t),Ze(e,be()),n;if(n===6)throw Error(A(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,In(e,Xe,Ht),Ze(e,be()),null}function Lu(e,t){var n=ae;ae|=1;try{return e(t)}finally{ae=n,ae===0&&(Er=be()+500,ko&&zn())}}function $n(e){cn!==null&&cn.tag===0&&!(ae&6)&&_r();var t=ae;ae|=1;var n=gt.transition,r=se;try{if(gt.transition=null,se=1,e)return e()}finally{se=r,gt.transition=n,ae=t,!(ae&6)&&zn()}}function Au(){nt=pr.current,me(pr)}function qn(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,_x(n)),Ce!==null)for(n=Ce.return;n!==null;){var r=n;switch(pu(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&$a();break;case 3:Cr(),me(Ye),me(Oe),wu();break;case 5:vu(r);break;case 4:Cr();break;case 13:me(ge);break;case 19:me(ge);break;case 10:gu(r.type._context);break;case 22:case 23:Au()}n=n.return}if(Te=e,Ce=e=_n(e.current,null),Ae=nt=t,Pe=0,Li=null,Nu=zo=Hn=0,Xe=ui=null,Mn!==null){for(t=0;t<Mn.length;t++)if(n=Mn[t],r=n.interleaved,r!==null){n.interleaved=null;var i=r.next,a=n.pending;if(a!==null){var o=a.next;a.next=i,r.next=o}n.pending=r}Mn=null}return e}function Am(e,t){do{var n=Ce;try{if(hu(),Pa.current=to,eo){for(var r=ye.memoizedState;r!==null;){var i=r.queue;i!==null&&(i.pending=null),r=r.next}eo=!1}if(Vn=0,Ne=Ee=ye=null,si=!1,Pi=0,Pu.current=null,n===null||n.return===null){Pe=1,Li=t,Ce=null;break}e:{var a=e,o=n.return,s=n,l=t;if(t=Ae,s.flags|=32768,l!==null&&typeof l=="object"&&typeof l.then=="function"){var u=l,d=s,c=d.tag;if(!(d.mode&1)&&(c===0||c===11||c===15)){var p=d.alternate;p?(d.updateQueue=p.updateQueue,d.memoizedState=p.memoizedState,d.lanes=p.lanes):(d.updateQueue=null,d.memoizedState=null)}var f=Jc(o);if(f!==null){f.flags&=-257,ed(f,o,s,a,t),f.mode&1&&Zc(a,u,t),t=f,l=u;var y=t.updateQueue;if(y===null){var _=new Set;_.add(l),t.updateQueue=_}else y.add(l);break e}else{if(!(t&1)){Zc(a,u,t),Ru();break e}l=Error(A(426))}}else if(he&&s.mode&1){var S=Jc(o);if(S!==null){!(S.flags&65536)&&(S.flags|=256),ed(S,o,s,a,t),fu(zr(l,s));break e}}a=l=zr(l,s),Pe!==4&&(Pe=2),ui===null?ui=[a]:ui.push(a),a=o;do{switch(a.tag){case 3:a.flags|=65536,t&=-t,a.lanes|=t;var m=hm(a,l,t);$c(a,m);break e;case 1:s=l;var h=a.type,g=a.stateNode;if(!(a.flags&128)&&(typeof h.getDerivedStateFromError=="function"||g!==null&&typeof g.componentDidCatch=="function"&&(yn===null||!yn.has(g)))){a.flags|=65536,t&=-t,a.lanes|=t;var C=gm(a,s,t);$c(a,C);break e}}a=a.return}while(a!==null)}Im(n)}catch(E){t=E,Ce===n&&n!==null&&(Ce=n=n.return);continue}break}while(!0)}function Rm(){var e=no.current;return no.current=to,e===null?to:e}function Ru(){(Pe===0||Pe===3||Pe===2)&&(Pe=4),Te===null||!(Hn&268435455)&&!(zo&268435455)||ln(Te,Ae)}function ao(e,t){var n=ae;ae|=2;var r=Rm();(Te!==e||Ae!==t)&&(Ht=null,qn(e,t));do try{Vx();break}catch(i){Am(e,i)}while(!0);if(hu(),ae=n,no.current=r,Ce!==null)throw Error(A(261));return Te=null,Ae=0,Pe}function Vx(){for(;Ce!==null;)jm(Ce)}function Hx(){for(;Ce!==null&&!gy();)jm(Ce)}function jm(e){var t=Fm(e.alternate,e,nt);e.memoizedProps=e.pendingProps,t===null?Im(e):Ce=t,Pu.current=null}function Im(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=Mx(n,t),n!==null){n.flags&=32767,Ce=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Pe=6,Ce=null;return}}else if(n=Fx(n,t,nt),n!==null){Ce=n;return}if(t=t.sibling,t!==null){Ce=t;return}Ce=t=e}while(t!==null);Pe===0&&(Pe=5)}function In(e,t,n){var r=se,i=gt.transition;try{gt.transition=null,se=1,$x(e,t,n,r)}finally{gt.transition=i,se=r}return null}function $x(e,t,n,r){do _r();while(cn!==null);if(ae&6)throw Error(A(327));n=e.finishedWork;var i=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(A(177));e.callbackNode=null,e.callbackPriority=0;var a=n.lanes|n.childLanes;if(zy(e,a),e===Te&&(Ce=Te=null,Ae=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||pa||(pa=!0,Mm(qa,function(){return _r(),null})),a=(n.flags&15990)!==0,n.subtreeFlags&15990||a){a=gt.transition,gt.transition=null;var o=se;se=1;var s=ae;ae|=4,Pu.current=null,qx(e,n),Nm(n,e),px(al),Ba=!!il,al=il=null,e.current=n,Wx(n),yy(),ae=s,se=o,gt.transition=a}else e.current=n;if(pa&&(pa=!1,cn=e,io=i),a=e.pendingLanes,a===0&&(yn=null),vy(n.stateNode),Ze(e,be()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)i=t[n],r(i.value,{componentStack:i.stack,digest:i.digest});if(ro)throw ro=!1,e=Cl,Cl=null,e;return io&1&&e.tag!==0&&_r(),a=e.pendingLanes,a&1?e===zl?ci++:(ci=0,zl=e):ci=0,zn(),null}function _r(){if(cn!==null){var e=hf(io),t=gt.transition,n=se;try{if(gt.transition=null,se=16>e?16:e,cn===null)var r=!1;else{if(e=cn,cn=null,io=0,ae&6)throw Error(A(331));var i=ae;for(ae|=4,U=e.current;U!==null;){var a=U,o=a.child;if(U.flags&16){var s=a.deletions;if(s!==null){for(var l=0;l<s.length;l++){var u=s[l];for(U=u;U!==null;){var d=U;switch(d.tag){case 0:case 11:case 15:li(8,d,a)}var c=d.child;if(c!==null)c.return=d,U=c;else for(;U!==null;){d=U;var p=d.sibling,f=d.return;if(zm(d),d===u){U=null;break}if(p!==null){p.return=f,U=p;break}U=f}}}var y=a.alternate;if(y!==null){var _=y.child;if(_!==null){y.child=null;do{var S=_.sibling;_.sibling=null,_=S}while(_!==null)}}U=a}}if(a.subtreeFlags&2064&&o!==null)o.return=a,U=o;else e:for(;U!==null;){if(a=U,a.flags&2048)switch(a.tag){case 0:case 11:case 15:li(9,a,a.return)}var m=a.sibling;if(m!==null){m.return=a.return,U=m;break e}U=a.return}}var h=e.current;for(U=h;U!==null;){o=U;var g=o.child;if(o.subtreeFlags&2064&&g!==null)g.return=o,U=g;else e:for(o=h;U!==null;){if(s=U,s.flags&2048)try{switch(s.tag){case 0:case 11:case 15:Co(9,s)}}catch(E){ve(s,s.return,E)}if(s===o){U=null;break e}var C=s.sibling;if(C!==null){C.return=s.return,U=C;break e}U=s.return}}if(ae=i,zn(),Mt&&typeof Mt.onPostCommitFiberRoot=="function")try{Mt.onPostCommitFiberRoot(yo,e)}catch{}r=!0}return r}finally{se=n,gt.transition=t}}return!1}function fd(e,t,n){t=zr(n,t),t=hm(e,t,1),e=gn(e,t,1),t=Ve(),e!==null&&(qi(e,1,t),Ze(e,t))}function ve(e,t,n){if(e.tag===3)fd(e,e,n);else for(;t!==null;){if(t.tag===3){fd(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(yn===null||!yn.has(r))){e=zr(n,e),e=gm(t,e,1),t=gn(t,e,1),e=Ve(),t!==null&&(qi(t,1,e),Ze(t,e));break}}t=t.return}}function Qx(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=Ve(),e.pingedLanes|=e.suspendedLanes&n,Te===e&&(Ae&n)===n&&(Pe===4||Pe===3&&(Ae&130023424)===Ae&&500>be()-Tu?qn(e,0):Nu|=n),Ze(e,t)}function Dm(e,t){t===0&&(e.mode&1?(t=na,na<<=1,!(na&130023424)&&(na=4194304)):t=1);var n=Ve();e=Zt(e,t),e!==null&&(qi(e,t,n),Ze(e,n))}function Xx(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),Dm(e,n)}function Kx(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(A(314))}r!==null&&r.delete(t),Dm(e,n)}var Fm;Fm=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||Ye.current)Ke=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return Ke=!1,Dx(e,t,n);Ke=!!(e.flags&131072)}else Ke=!1,he&&t.flags&1048576&&Wf(t,Ka,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;Ta(e,t),e=t.pendingProps;var i=kr(t,Oe.current);xr(t,n),i=bu(null,t,r,e,i,n);var a=Su();return t.flags|=1,typeof i=="object"&&i!==null&&typeof i.render=="function"&&i.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,Ge(r)?(a=!0,Qa(t)):a=!1,t.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,xu(t),i.updater=So,t.stateNode=i,i._reactInternals=t,ml(t,r,e,n),t=yl(null,t,r,!0,a,n)):(t.tag=0,he&&a&&du(t),Be(null,t,i,n),t=t.child),t;case 16:r=t.elementType;e:{switch(Ta(e,t),e=t.pendingProps,i=r._init,r=i(r._payload),t.type=r,i=t.tag=Gx(r),e=Ct(r,e),i){case 0:t=gl(null,t,r,e,n);break e;case 1:t=rd(null,t,r,e,n);break e;case 11:t=td(null,t,r,e,n);break e;case 14:t=nd(null,t,r,Ct(r.type,e),n);break e}throw Error(A(306,r,""))}return t;case 0:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Ct(r,i),gl(e,t,r,i,n);case 1:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Ct(r,i),rd(e,t,r,i,n);case 3:e:{if(vm(t),e===null)throw Error(A(387));r=t.pendingProps,a=t.memoizedState,i=a.element,Qf(e,t),Za(t,r,null,n);var o=t.memoizedState;if(r=o.element,a.isDehydrated)if(a={element:r,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},t.updateQueue.baseState=a,t.memoizedState=a,t.flags&256){i=zr(Error(A(423)),t),t=id(e,t,r,n,i);break e}else if(r!==i){i=zr(Error(A(424)),t),t=id(e,t,r,n,i);break e}else for(it=hn(t.stateNode.containerInfo.firstChild),at=t,he=!0,Et=null,n=Hf(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(br(),r===i){t=Jt(e,t,n);break e}Be(e,t,r,n)}t=t.child}return t;case 5:return Xf(t),e===null&&dl(t),r=t.type,i=t.pendingProps,a=e!==null?e.memoizedProps:null,o=i.children,ol(r,i)?o=null:a!==null&&ol(r,a)&&(t.flags|=32),_m(e,t),Be(e,t,o,n),t.child;case 6:return e===null&&dl(t),null;case 13:return wm(e,t,n);case 4:return _u(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=Sr(t,null,r,n):Be(e,t,r,n),t.child;case 11:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Ct(r,i),td(e,t,r,i,n);case 7:return Be(e,t,t.pendingProps,n),t.child;case 8:return Be(e,t,t.pendingProps.children,n),t.child;case 12:return Be(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,i=t.pendingProps,a=t.memoizedProps,o=i.value,pe(Ya,r._currentValue),r._currentValue=o,a!==null)if(Lt(a.value,o)){if(a.children===i.children&&!Ye.current){t=Jt(e,t,n);break e}}else for(a=t.child,a!==null&&(a.return=t);a!==null;){var s=a.dependencies;if(s!==null){o=a.child;for(var l=s.firstContext;l!==null;){if(l.context===r){if(a.tag===1){l=Kt(-1,n&-n),l.tag=2;var u=a.updateQueue;if(u!==null){u=u.shared;var d=u.pending;d===null?l.next=l:(l.next=d.next,d.next=l),u.pending=l}}a.lanes|=n,l=a.alternate,l!==null&&(l.lanes|=n),pl(a.return,n,t),s.lanes|=n;break}l=l.next}}else if(a.tag===10)o=a.type===t.type?null:a.child;else if(a.tag===18){if(o=a.return,o===null)throw Error(A(341));o.lanes|=n,s=o.alternate,s!==null&&(s.lanes|=n),pl(o,n,t),o=a.sibling}else o=a.child;if(o!==null)o.return=a;else for(o=a;o!==null;){if(o===t){o=null;break}if(a=o.sibling,a!==null){a.return=o.return,o=a;break}o=o.return}a=o}Be(e,t,i.children,n),t=t.child}return t;case 9:return i=t.type,r=t.pendingProps.children,xr(t,n),i=yt(i),r=r(i),t.flags|=1,Be(e,t,r,n),t.child;case 14:return r=t.type,i=Ct(r,t.pendingProps),i=Ct(r.type,i),nd(e,t,r,i,n);case 15:return ym(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Ct(r,i),Ta(e,t),t.tag=1,Ge(r)?(e=!0,Qa(t)):e=!1,xr(t,n),mm(t,r,i),ml(t,r,i,n),yl(null,t,r,!0,e,n);case 19:return km(e,t,n);case 22:return xm(e,t,n)}throw Error(A(156,t.tag))};function Mm(e,t){return df(e,t)}function Yx(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function ht(e,t,n,r){return new Yx(e,t,n,r)}function ju(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Gx(e){if(typeof e=="function")return ju(e)?1:0;if(e!=null){if(e=e.$$typeof,e===Jl)return 11;if(e===eu)return 14}return 2}function _n(e,t){var n=e.alternate;return n===null?(n=ht(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function Ra(e,t,n,r,i,a){var o=2;if(r=e,typeof e=="function")ju(e)&&(o=1);else if(typeof e=="string")o=5;else e:switch(e){case nr:return Wn(n.children,i,a,t);case Zl:o=8,i|=8;break;case Ms:return e=ht(12,n,t,i|2),e.elementType=Ms,e.lanes=a,e;case Os:return e=ht(13,n,t,i),e.elementType=Os,e.lanes=a,e;case qs:return e=ht(19,n,t,i),e.elementType=qs,e.lanes=a,e;case Qp:return Eo(n,i,a,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case Hp:o=10;break e;case $p:o=9;break e;case Jl:o=11;break e;case eu:o=14;break e;case an:o=16,r=null;break e}throw Error(A(130,e==null?e:typeof e,""))}return t=ht(o,n,t,i),t.elementType=e,t.type=r,t.lanes=a,t}function Wn(e,t,n,r){return e=ht(7,e,r,t),e.lanes=n,e}function Eo(e,t,n,r){return e=ht(22,e,r,t),e.elementType=Qp,e.lanes=n,e.stateNode={isHidden:!1},e}function ps(e,t,n){return e=ht(6,e,null,t),e.lanes=n,e}function fs(e,t,n){return t=ht(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function Zx(e,t,n,r,i){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=Qo(0),this.expirationTimes=Qo(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Qo(0),this.identifierPrefix=r,this.onRecoverableError=i,this.mutableSourceEagerHydrationData=null}function Iu(e,t,n,r,i,a,o,s,l){return e=new Zx(e,t,n,s,l),t===1?(t=1,a===!0&&(t|=8)):t=0,a=ht(3,null,null,t),e.current=a,a.stateNode=e,a.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},xu(a),e}function Jx(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:tr,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function Om(e){if(!e)return bn;e=e._reactInternals;e:{if(Kn(e)!==e||e.tag!==1)throw Error(A(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(Ge(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(A(171))}if(e.tag===1){var n=e.type;if(Ge(n))return Of(e,n,t)}return t}function qm(e,t,n,r,i,a,o,s,l){return e=Iu(n,r,!0,e,i,a,o,s,l),e.context=Om(null),n=e.current,r=Ve(),i=xn(n),a=Kt(r,i),a.callback=t??null,gn(n,a,i),e.current.lanes=i,qi(e,i,r),Ze(e,r),e}function Po(e,t,n,r){var i=t.current,a=Ve(),o=xn(i);return n=Om(n),t.context===null?t.context=n:t.pendingContext=n,t=Kt(a,o),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=gn(i,t,o),e!==null&&(Nt(e,i,o,a),Ea(e,i,o)),o}function oo(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function md(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function Du(e,t){md(e,t),(e=e.alternate)&&md(e,t)}function e0(){return null}var Wm=typeof reportError=="function"?reportError:function(e){console.error(e)};function Fu(e){this._internalRoot=e}No.prototype.render=Fu.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(A(409));Po(e,t,null,null)};No.prototype.unmount=Fu.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;$n(function(){Po(null,e,null,null)}),t[Gt]=null}};function No(e){this._internalRoot=e}No.prototype.unstable_scheduleHydration=function(e){if(e){var t=xf();e={blockedOn:null,target:e,priority:t};for(var n=0;n<sn.length&&t!==0&&t<sn[n].priority;n++);sn.splice(n,0,e),n===0&&vf(e)}};function Mu(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function To(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function hd(){}function t0(e,t,n,r,i){if(i){if(typeof r=="function"){var a=r;r=function(){var u=oo(o);a.call(u)}}var o=qm(t,r,e,0,null,!1,!1,"",hd);return e._reactRootContainer=o,e[Gt]=o.current,bi(e.nodeType===8?e.parentNode:e),$n(),o}for(;i=e.lastChild;)e.removeChild(i);if(typeof r=="function"){var s=r;r=function(){var u=oo(l);s.call(u)}}var l=Iu(e,0,!1,null,null,!1,!1,"",hd);return e._reactRootContainer=l,e[Gt]=l.current,bi(e.nodeType===8?e.parentNode:e),$n(function(){Po(t,l,n,r)}),l}function Lo(e,t,n,r,i){var a=n._reactRootContainer;if(a){var o=a;if(typeof i=="function"){var s=i;i=function(){var l=oo(o);s.call(l)}}Po(t,o,e,i)}else o=t0(n,t,e,i,r);return oo(o)}gf=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=Zr(t.pendingLanes);n!==0&&(ru(t,n|1),Ze(t,be()),!(ae&6)&&(Er=be()+500,zn()))}break;case 13:$n(function(){var r=Zt(e,1);if(r!==null){var i=Ve();Nt(r,e,1,i)}}),Du(e,1)}};iu=function(e){if(e.tag===13){var t=Zt(e,134217728);if(t!==null){var n=Ve();Nt(t,e,134217728,n)}Du(e,134217728)}};yf=function(e){if(e.tag===13){var t=xn(e),n=Zt(e,t);if(n!==null){var r=Ve();Nt(n,e,t,r)}Du(e,t)}};xf=function(){return se};_f=function(e,t){var n=se;try{return se=e,t()}finally{se=n}};Ys=function(e,t,n){switch(t){case"input":if(Us(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=wo(r);if(!i)throw Error(A(90));Kp(r),Us(r,i)}}}break;case"textarea":Gp(e,n);break;case"select":t=n.value,t!=null&&mr(e,!!n.multiple,t,!1)}};af=Lu;of=$n;var n0={usingClientEntryPoint:!1,Events:[Bi,or,wo,nf,rf,Lu]},$r={findFiberByHostInstance:Fn,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},r0={bundleType:$r.bundleType,version:$r.version,rendererPackageName:$r.rendererPackageName,rendererConfig:$r.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:en.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=uf(e),e===null?null:e.stateNode},findFiberByHostInstance:$r.findFiberByHostInstance||e0,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var fa=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!fa.isDisabled&&fa.supportsFiber)try{yo=fa.inject(r0),Mt=fa}catch{}}lt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=n0;lt.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!Mu(t))throw Error(A(200));return Jx(e,t,null,n)};lt.createRoot=function(e,t){if(!Mu(e))throw Error(A(299));var n=!1,r="",i=Wm;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(i=t.onRecoverableError)),t=Iu(e,1,!1,null,null,n,!1,r,i),e[Gt]=t.current,bi(e.nodeType===8?e.parentNode:e),new Fu(t)};lt.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(A(188)):(e=Object.keys(e).join(","),Error(A(268,e)));return e=uf(t),e=e===null?null:e.stateNode,e};lt.flushSync=function(e){return $n(e)};lt.hydrate=function(e,t,n){if(!To(t))throw Error(A(200));return Lo(null,e,t,!0,n)};lt.hydrateRoot=function(e,t,n){if(!Mu(e))throw Error(A(405));var r=n!=null&&n.hydratedSources||null,i=!1,a="",o=Wm;if(n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(a=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),t=qm(t,null,e,1,n??null,i,!1,a,o),e[Gt]=t.current,bi(e),r)for(e=0;e<r.length;e++)n=r[e],i=n._getVersion,i=i(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,i]:t.mutableSourceEagerHydrationData.push(n,i);return new No(t)};lt.render=function(e,t,n){if(!To(t))throw Error(A(200));return Lo(null,e,t,!1,n)};lt.unmountComponentAtNode=function(e){if(!To(e))throw Error(A(40));return e._reactRootContainer?($n(function(){Lo(null,null,e,!1,function(){e._reactRootContainer=null,e[Gt]=null})}),!0):!1};lt.unstable_batchedUpdates=Lu;lt.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!To(n))throw Error(A(200));if(e==null||e._reactInternals===void 0)throw Error(A(38));return Lo(e,t,n,!1,r)};lt.version="18.3.1-next-f1338f8080-20240426";function Bm(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Bm)}catch(e){console.error(e)}}Bm(),Wp.exports=lt;var i0=Wp.exports,gd=i0;Ds.createRoot=gd.createRoot,Ds.hydrateRoot=gd.hydrateRoot;/**
 * @remix-run/router v1.23.2
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ai(){return Ai=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ai.apply(this,arguments)}var dn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(dn||(dn={}));const yd="popstate";function a0(e){e===void 0&&(e={});function t(r,i){let{pathname:a,search:o,hash:s}=r.location;return Nl("",{pathname:a,search:o,hash:s},i.state&&i.state.usr||null,i.state&&i.state.key||"default")}function n(r,i){return typeof i=="string"?i:so(i)}return s0(t,n,null,e)}function we(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function Ou(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function o0(){return Math.random().toString(36).substr(2,8)}function xd(e,t){return{usr:e.state,key:e.key,idx:t}}function Nl(e,t,n,r){return n===void 0&&(n=null),Ai({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?jr(t):t,{state:n,key:t&&t.key||r||o0()})}function so(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function jr(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function s0(e,t,n,r){r===void 0&&(r={});let{window:i=document.defaultView,v5Compat:a=!1}=r,o=i.history,s=dn.Pop,l=null,u=d();u==null&&(u=0,o.replaceState(Ai({},o.state,{idx:u}),""));function d(){return(o.state||{idx:null}).idx}function c(){s=dn.Pop;let S=d(),m=S==null?null:S-u;u=S,l&&l({action:s,location:_.location,delta:m})}function p(S,m){s=dn.Push;let h=Nl(_.location,S,m);u=d()+1;let g=xd(h,u),C=_.createHref(h);try{o.pushState(g,"",C)}catch(E){if(E instanceof DOMException&&E.name==="DataCloneError")throw E;i.location.assign(C)}a&&l&&l({action:s,location:_.location,delta:1})}function f(S,m){s=dn.Replace;let h=Nl(_.location,S,m);u=d();let g=xd(h,u),C=_.createHref(h);o.replaceState(g,"",C),a&&l&&l({action:s,location:_.location,delta:0})}function y(S){let m=i.location.origin!=="null"?i.location.origin:i.location.href,h=typeof S=="string"?S:so(S);return h=h.replace(/ $/,"%20"),we(m,"No window.location.(origin|href) available to create URL for href: "+h),new URL(h,m)}let _={get action(){return s},get location(){return e(i,o)},listen(S){if(l)throw new Error("A history only accepts one active listener");return i.addEventListener(yd,c),l=S,()=>{i.removeEventListener(yd,c),l=null}},createHref(S){return t(i,S)},createURL:y,encodeLocation(S){let m=y(S);return{pathname:m.pathname,search:m.search,hash:m.hash}},push:p,replace:f,go(S){return o.go(S)}};return _}var _d;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(_d||(_d={}));function l0(e,t,n){return n===void 0&&(n="/"),u0(e,t,n)}function u0(e,t,n,r){let i=typeof t=="string"?jr(t):t,a=Pr(i.pathname||"/",n);if(a==null)return null;let o=Um(e);c0(o);let s=null;for(let l=0;s==null&&l<o.length;++l){let u=w0(a);s=_0(o[l],u)}return s}function Um(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let i=(a,o,s)=>{let l={relativePath:s===void 0?a.path||"":s,caseSensitive:a.caseSensitive===!0,childrenIndex:o,route:a};l.relativePath.startsWith("/")&&(we(l.relativePath.startsWith(r),'Absolute route path "'+l.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),l.relativePath=l.relativePath.slice(r.length));let u=vn([r,l.relativePath]),d=n.concat(l);a.children&&a.children.length>0&&(we(a.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),Um(a.children,t,d,u)),!(a.path==null&&!a.index)&&t.push({path:u,score:y0(u,a.index),routesMeta:d})};return e.forEach((a,o)=>{var s;if(a.path===""||!((s=a.path)!=null&&s.includes("?")))i(a,o);else for(let l of Vm(a.path))i(a,o,l)}),t}function Vm(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,i=n.endsWith("?"),a=n.replace(/\?$/,"");if(r.length===0)return i?[a,""]:[a];let o=Vm(r.join("/")),s=[];return s.push(...o.map(l=>l===""?a:[a,l].join("/"))),i&&s.push(...o),s.map(l=>e.startsWith("/")&&l===""?"/":l)}function c0(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:x0(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const d0=/^:[\w-]+$/,p0=3,f0=2,m0=1,h0=10,g0=-2,vd=e=>e==="*";function y0(e,t){let n=e.split("/"),r=n.length;return n.some(vd)&&(r+=g0),t&&(r+=f0),n.filter(i=>!vd(i)).reduce((i,a)=>i+(d0.test(a)?p0:a===""?m0:h0),r)}function x0(e,t){return e.length===t.length&&e.slice(0,-1).every((r,i)=>r===t[i])?e[e.length-1]-t[t.length-1]:0}function _0(e,t,n){let{routesMeta:r}=e,i={},a="/",o=[];for(let s=0;s<r.length;++s){let l=r[s],u=s===r.length-1,d=a==="/"?t:t.slice(a.length)||"/",c=Tl({path:l.relativePath,caseSensitive:l.caseSensitive,end:u},d),p=l.route;if(!c)return null;Object.assign(i,c.params),o.push({params:i,pathname:vn([a,c.pathname]),pathnameBase:z0(vn([a,c.pathnameBase])),route:p}),c.pathnameBase!=="/"&&(a=vn([a,c.pathnameBase]))}return o}function Tl(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=v0(e.path,e.caseSensitive,e.end),i=t.match(n);if(!i)return null;let a=i[0],o=a.replace(/(.)\/+$/,"$1"),s=i.slice(1);return{params:r.reduce((u,d,c)=>{let{paramName:p,isOptional:f}=d;if(p==="*"){let _=s[c]||"";o=a.slice(0,a.length-_.length).replace(/(.)\/+$/,"$1")}const y=s[c];return f&&!y?u[p]=void 0:u[p]=(y||"").replace(/%2F/g,"/"),u},{}),pathname:a,pathnameBase:o,pattern:e}}function v0(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),Ou(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],i="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(o,s,l)=>(r.push({paramName:s,isOptional:l!=null}),l?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),i+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?i+="\\/*$":e!==""&&e!=="/"&&(i+="(?:(?=\\/|$))"),[new RegExp(i,t?void 0:"i"),r]}function w0(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return Ou(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function Pr(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const k0=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,b0=e=>k0.test(e);function S0(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:i=""}=typeof e=="string"?jr(e):e,a;if(n)if(b0(n))a=n;else{if(n.includes("//")){let o=n;n=n.replace(/\/\/+/g,"/"),Ou(!1,"Pathnames cannot have embedded double slashes - normalizing "+(o+" -> "+n))}n.startsWith("/")?a=wd(n.substring(1),"/"):a=wd(n,t)}else a=t;return{pathname:a,search:E0(r),hash:P0(i)}}function wd(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(i=>{i===".."?n.length>1&&n.pop():i!=="."&&n.push(i)}),n.length>1?n.join("/"):"/"}function ms(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function C0(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function Hm(e,t){let n=C0(e);return t?n.map((r,i)=>i===n.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function $m(e,t,n,r){r===void 0&&(r=!1);let i;typeof e=="string"?i=jr(e):(i=Ai({},e),we(!i.pathname||!i.pathname.includes("?"),ms("?","pathname","search",i)),we(!i.pathname||!i.pathname.includes("#"),ms("#","pathname","hash",i)),we(!i.search||!i.search.includes("#"),ms("#","search","hash",i)));let a=e===""||i.pathname==="",o=a?"/":i.pathname,s;if(o==null)s=n;else{let c=t.length-1;if(!r&&o.startsWith("..")){let p=o.split("/");for(;p[0]==="..";)p.shift(),c-=1;i.pathname=p.join("/")}s=c>=0?t[c]:"/"}let l=S0(i,s),u=o&&o!=="/"&&o.endsWith("/"),d=(a||o===".")&&n.endsWith("/");return!l.pathname.endsWith("/")&&(u||d)&&(l.pathname+="/"),l}const vn=e=>e.join("/").replace(/\/\/+/g,"/"),z0=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),E0=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,P0=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function N0(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Qm=["post","put","patch","delete"];new Set(Qm);const T0=["get",...Qm];new Set(T0);/**
 * React Router v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ri(){return Ri=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ri.apply(this,arguments)}const Ao=z.createContext(null),Xm=z.createContext(null),En=z.createContext(null),Ro=z.createContext(null),Pn=z.createContext({outlet:null,matches:[],isDataRoute:!1}),Km=z.createContext(null);function L0(e,t){let{relative:n}=t===void 0?{}:t;Vi()||we(!1);let{basename:r,navigator:i}=z.useContext(En),{hash:a,pathname:o,search:s}=jo(e,{relative:n}),l=o;return r!=="/"&&(l=o==="/"?r:vn([r,o])),i.createHref({pathname:l,search:s,hash:a})}function Vi(){return z.useContext(Ro)!=null}function Hi(){return Vi()||we(!1),z.useContext(Ro).location}function Ym(e){z.useContext(En).static||z.useLayoutEffect(e)}function A0(){let{isDataRoute:e}=z.useContext(Pn);return e?H0():R0()}function R0(){Vi()||we(!1);let e=z.useContext(Ao),{basename:t,future:n,navigator:r}=z.useContext(En),{matches:i}=z.useContext(Pn),{pathname:a}=Hi(),o=JSON.stringify(Hm(i,n.v7_relativeSplatPath)),s=z.useRef(!1);return Ym(()=>{s.current=!0}),z.useCallback(function(u,d){if(d===void 0&&(d={}),!s.current)return;if(typeof u=="number"){r.go(u);return}let c=$m(u,JSON.parse(o),a,d.relative==="path");e==null&&t!=="/"&&(c.pathname=c.pathname==="/"?t:vn([t,c.pathname])),(d.replace?r.replace:r.push)(c,d.state,d)},[t,r,o,a,e])}function Gm(){let{matches:e}=z.useContext(Pn),t=e[e.length-1];return t?t.params:{}}function jo(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=z.useContext(En),{matches:i}=z.useContext(Pn),{pathname:a}=Hi(),o=JSON.stringify(Hm(i,r.v7_relativeSplatPath));return z.useMemo(()=>$m(e,JSON.parse(o),a,n==="path"),[e,o,a,n])}function j0(e,t){return I0(e,t)}function I0(e,t,n,r){Vi()||we(!1);let{navigator:i}=z.useContext(En),{matches:a}=z.useContext(Pn),o=a[a.length-1],s=o?o.params:{};o&&o.pathname;let l=o?o.pathnameBase:"/";o&&o.route;let u=Hi(),d;if(t){var c;let S=typeof t=="string"?jr(t):t;l==="/"||(c=S.pathname)!=null&&c.startsWith(l)||we(!1),d=S}else d=u;let p=d.pathname||"/",f=p;if(l!=="/"){let S=l.replace(/^\//,"").split("/");f="/"+p.replace(/^\//,"").split("/").slice(S.length).join("/")}let y=l0(e,{pathname:f}),_=q0(y&&y.map(S=>Object.assign({},S,{params:Object.assign({},s,S.params),pathname:vn([l,i.encodeLocation?i.encodeLocation(S.pathname).pathname:S.pathname]),pathnameBase:S.pathnameBase==="/"?l:vn([l,i.encodeLocation?i.encodeLocation(S.pathnameBase).pathname:S.pathnameBase])})),a,n,r);return t&&_?z.createElement(Ro.Provider,{value:{location:Ri({pathname:"/",search:"",hash:"",state:null,key:"default"},d),navigationType:dn.Pop}},_):_}function D0(){let e=V0(),t=N0(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,i={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return z.createElement(z.Fragment,null,z.createElement("h2",null,"Unexpected Application Error!"),z.createElement("h3",{style:{fontStyle:"italic"}},t),n?z.createElement("pre",{style:i},n):null,null)}const F0=z.createElement(D0,null);class M0 extends z.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?z.createElement(Pn.Provider,{value:this.props.routeContext},z.createElement(Km.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function O0(e){let{routeContext:t,match:n,children:r}=e,i=z.useContext(Ao);return i&&i.static&&i.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(i.staticContext._deepestRenderedBoundaryId=n.route.id),z.createElement(Pn.Provider,{value:t},r)}function q0(e,t,n,r){var i;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var a;if(!n)return null;if(n.errors)e=n.matches;else if((a=r)!=null&&a.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let o=e,s=(i=n)==null?void 0:i.errors;if(s!=null){let d=o.findIndex(c=>c.route.id&&(s==null?void 0:s[c.route.id])!==void 0);d>=0||we(!1),o=o.slice(0,Math.min(o.length,d+1))}let l=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let d=0;d<o.length;d++){let c=o[d];if((c.route.HydrateFallback||c.route.hydrateFallbackElement)&&(u=d),c.route.id){let{loaderData:p,errors:f}=n,y=c.route.loader&&p[c.route.id]===void 0&&(!f||f[c.route.id]===void 0);if(c.route.lazy||y){l=!0,u>=0?o=o.slice(0,u+1):o=[o[0]];break}}}return o.reduceRight((d,c,p)=>{let f,y=!1,_=null,S=null;n&&(f=s&&c.route.id?s[c.route.id]:void 0,_=c.route.errorElement||F0,l&&(u<0&&p===0?($0("route-fallback"),y=!0,S=null):u===p&&(y=!0,S=c.route.hydrateFallbackElement||null)));let m=t.concat(o.slice(0,p+1)),h=()=>{let g;return f?g=_:y?g=S:c.route.Component?g=z.createElement(c.route.Component,null):c.route.element?g=c.route.element:g=d,z.createElement(O0,{match:c,routeContext:{outlet:d,matches:m,isDataRoute:n!=null},children:g})};return n&&(c.route.ErrorBoundary||c.route.errorElement||p===0)?z.createElement(M0,{location:n.location,revalidation:n.revalidation,component:_,error:f,children:h(),routeContext:{outlet:null,matches:m,isDataRoute:!0}}):h()},null)}var Zm=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(Zm||{}),Jm=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Jm||{});function W0(e){let t=z.useContext(Ao);return t||we(!1),t}function B0(e){let t=z.useContext(Xm);return t||we(!1),t}function U0(e){let t=z.useContext(Pn);return t||we(!1),t}function eh(e){let t=U0(),n=t.matches[t.matches.length-1];return n.route.id||we(!1),n.route.id}function V0(){var e;let t=z.useContext(Km),n=B0(),r=eh();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function H0(){let{router:e}=W0(Zm.UseNavigateStable),t=eh(Jm.UseNavigateStable),n=z.useRef(!1);return Ym(()=>{n.current=!0}),z.useCallback(function(i,a){a===void 0&&(a={}),n.current&&(typeof i=="number"?e.navigate(i):e.navigate(i,Ri({fromRouteId:t},a)))},[e,t])}const kd={};function $0(e,t,n){kd[e]||(kd[e]=!0)}function Q0(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function ei(e){we(!1)}function X0(e){let{basename:t="/",children:n=null,location:r,navigationType:i=dn.Pop,navigator:a,static:o=!1,future:s}=e;Vi()&&we(!1);let l=t.replace(/^\/*/,"/"),u=z.useMemo(()=>({basename:l,navigator:a,static:o,future:Ri({v7_relativeSplatPath:!1},s)}),[l,s,a,o]);typeof r=="string"&&(r=jr(r));let{pathname:d="/",search:c="",hash:p="",state:f=null,key:y="default"}=r,_=z.useMemo(()=>{let S=Pr(d,l);return S==null?null:{location:{pathname:S,search:c,hash:p,state:f,key:y},navigationType:i}},[l,d,c,p,f,y,i]);return _==null?null:z.createElement(En.Provider,{value:u},z.createElement(Ro.Provider,{children:n,value:_}))}function K0(e){let{children:t,location:n}=e;return j0(Ll(t),n)}new Promise(()=>{});function Ll(e,t){t===void 0&&(t=[]);let n=[];return z.Children.forEach(e,(r,i)=>{if(!z.isValidElement(r))return;let a=[...t,i];if(r.type===z.Fragment){n.push.apply(n,Ll(r.props.children,a));return}r.type!==ei&&we(!1),!r.props.index||!r.props.children||we(!1);let o={id:r.props.id||a.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(o.children=Ll(r.props.children,a)),n.push(o)}),n}/**
 * React Router DOM v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function lo(){return lo=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},lo.apply(this,arguments)}function th(e,t){if(e==null)return{};var n={},r=Object.keys(e),i,a;for(a=0;a<r.length;a++)i=r[a],!(t.indexOf(i)>=0)&&(n[i]=e[i]);return n}function Y0(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function G0(e,t){return e.button===0&&(!t||t==="_self")&&!Y0(e)}const Z0=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","viewTransition"],J0=["aria-current","caseSensitive","className","end","style","to","viewTransition","children"],e1="6";try{window.__reactRouterVersion=e1}catch{}const t1=z.createContext({isTransitioning:!1}),n1="startTransition",bd=Qg[n1];function r1(e){let{basename:t,children:n,future:r,window:i}=e,a=z.useRef();a.current==null&&(a.current=a0({window:i,v5Compat:!0}));let o=a.current,[s,l]=z.useState({action:o.action,location:o.location}),{v7_startTransition:u}=r||{},d=z.useCallback(c=>{u&&bd?bd(()=>l(c)):l(c)},[l,u]);return z.useLayoutEffect(()=>o.listen(d),[o,d]),z.useEffect(()=>Q0(r),[r]),z.createElement(X0,{basename:t,children:n,location:s.location,navigationType:s.action,navigator:o,future:r})}const i1=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",a1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,qt=z.forwardRef(function(t,n){let{onClick:r,relative:i,reloadDocument:a,replace:o,state:s,target:l,to:u,preventScrollReset:d,viewTransition:c}=t,p=th(t,Z0),{basename:f}=z.useContext(En),y,_=!1;if(typeof u=="string"&&a1.test(u)&&(y=u,i1))try{let g=new URL(window.location.href),C=u.startsWith("//")?new URL(g.protocol+u):new URL(u),E=Pr(C.pathname,f);C.origin===g.origin&&E!=null?u=E+C.search+C.hash:_=!0}catch{}let S=L0(u,{relative:i}),m=l1(u,{replace:o,state:s,target:l,preventScrollReset:d,relative:i,viewTransition:c});function h(g){r&&r(g),g.defaultPrevented||m(g)}return z.createElement("a",lo({},p,{href:y||S,onClick:_||a?r:h,ref:n,target:l}))}),o1=z.forwardRef(function(t,n){let{"aria-current":r="page",caseSensitive:i=!1,className:a="",end:o=!1,style:s,to:l,viewTransition:u,children:d}=t,c=th(t,J0),p=jo(l,{relative:c.relative}),f=Hi(),y=z.useContext(Xm),{navigator:_,basename:S}=z.useContext(En),m=y!=null&&u1(p)&&u===!0,h=_.encodeLocation?_.encodeLocation(p).pathname:p.pathname,g=f.pathname,C=y&&y.navigation&&y.navigation.location?y.navigation.location.pathname:null;i||(g=g.toLowerCase(),C=C?C.toLowerCase():null,h=h.toLowerCase()),C&&S&&(C=Pr(C,S)||C);const E=h!=="/"&&h.endsWith("/")?h.length-1:h.length;let b=g===h||!o&&g.startsWith(h)&&g.charAt(E)==="/",P=C!=null&&(C===h||!o&&C.startsWith(h)&&C.charAt(h.length)==="/"),T={isActive:b,isPending:P,isTransitioning:m},D=b?r:void 0,k;typeof a=="function"?k=a(T):k=[a,b?"active":null,P?"pending":null,m?"transitioning":null].filter(Boolean).join(" ");let I=typeof s=="function"?s(T):s;return z.createElement(qt,lo({},c,{"aria-current":D,className:k,ref:n,style:I,to:l,viewTransition:u}),typeof d=="function"?d(T):d)});var Al;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(Al||(Al={}));var Sd;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(Sd||(Sd={}));function s1(e){let t=z.useContext(Ao);return t||we(!1),t}function l1(e,t){let{target:n,replace:r,state:i,preventScrollReset:a,relative:o,viewTransition:s}=t===void 0?{}:t,l=A0(),u=Hi(),d=jo(e,{relative:o});return z.useCallback(c=>{if(G0(c,n)){c.preventDefault();let p=r!==void 0?r:so(u)===so(d);l(e,{replace:p,state:i,preventScrollReset:a,relative:o,viewTransition:s})}},[u,l,d,r,i,n,e,a,o,s])}function u1(e,t){t===void 0&&(t={});let n=z.useContext(t1);n==null&&we(!1);let{basename:r}=s1(Al.useViewTransitionState),i=jo(e,{relative:t.relative});if(!n.isTransitioning)return!1;let a=Pr(n.currentLocation.pathname,r)||n.currentLocation.pathname,o=Pr(n.nextLocation.pathname,r)||n.nextLocation.pathname;return Tl(i.pathname,o)!=null||Tl(i.pathname,a)!=null}const hs="ml-interview-progress",nh=z.createContext(null);function c1({children:e}){const[t,n]=z.useState(()=>{const u=localStorage.getItem(hs);return u?JSON.parse(u):{}});z.useEffect(()=>{localStorage.setItem(hs,JSON.stringify(t))},[t]);const r=z.useCallback((u,d,c)=>{n(p=>{var f;return{...p,[u]:{...p[u],[d]:{...(f=p[u])==null?void 0:f[d],status:c,lastAttempt:new Date().toISOString()}}}})},[]),i=z.useCallback((u,d,c)=>{n(p=>{var f,y,_;return{...p,[u]:{...p[u],[d]:{...(f=p[u])==null?void 0:f[d],status:((_=(y=p[u])==null?void 0:y[d])==null?void 0:_.status)||"in_progress",code:c,lastAttempt:new Date().toISOString()}}}})},[]),a=z.useCallback((u,d)=>{var p;const c=(p=t[u])==null?void 0:p[d];return{status:(c==null?void 0:c.status)||"not_started",code:c==null?void 0:c.code}},[t]),o=z.useCallback((u,d)=>{const c=t[u];if(!c||d===0)return 0;const p=Object.values(c).filter(f=>f.status==="completed").length;return Math.round(p/d*100)},[t]),s=z.useCallback(u=>{const d=u.reduce((p,f)=>p+f.problemCount,0);if(d===0)return 0;let c=0;for(const p of u){const f=t[p.id];f&&(c+=Object.values(f).filter(y=>y.status==="completed").length)}return Math.round(c/d*100)},[t]),l=z.useCallback(()=>{n({}),localStorage.removeItem(hs)},[]);return x.jsx(nh.Provider,{value:{progress:t,updateProblemStatus:r,saveProblemCode:i,getProblemProgress:a,getSectionProgress:o,getOverallProgress:s,resetProgress:l},children:e})}function $i(){const e=z.useContext(nh);if(!e)throw new Error("useProgress must be used within a ProgressProvider");return e}const Nr=[{id:"numpy-fundamentals",title:"NumPy Fundamentals",description:"Master NumPy array operations essential for ML development.",icon:"🔢",introduction:"\n# NumPy Fundamentals\n\nNumPy is the backbone of scientific computing in Python. Every ML library builds on it.\n\n## Key Concepts\n\n### Array Creation\n- `np.array()`: Create from lists\n- `np.zeros()`, `np.ones()`: Initialize with values\n- `np.arange()`, `np.linspace()`: Numeric ranges\n- `np.eye()`: Identity matrix\n\n### Indexing & Slicing\n- Basic: `arr[0]`, `arr[1:5]`\n- Advanced: `arr[[0, 2, 4]]`, `arr[arr > 0]`\n- Multi-dimensional: `arr[1, :]`, `arr[:, 2:4]`\n\n### Broadcasting\nArrays of different shapes can be combined:\n- (3, 4) + (4,) → broadcasts to (3, 4)\n- (3, 1) * (1, 4) → broadcasts to (3, 4)\n\n### Aggregations\n- `sum()`, `mean()`, `std()` - with axis parameter\n- `argmax()`, `argmin()` - find indices\n\n### Shape Manipulation\n- `reshape()`: Change dimensions\n- `transpose()`, `.T`: Swap axes\n- `flatten()`, `ravel()`: To 1D\n\nMaster these operations to write efficient ML code!\n    ",problems:["numpy-array-creation","numpy-indexing","numpy-broadcasting","numpy-aggregations","numpy-reshape-transpose"]},{id:"python-basics",title:"Python Basics for ML",description:"Master NumPy arrays and essential Python operations for machine learning.",icon:"🐍",introduction:"\n# Python Basics for Machine Learning\n\nNumPy is the foundation of nearly all machine learning in Python. Before diving into complex algorithms, you need to master array operations.\n\n## Key Concepts\n\n### NumPy Arrays\n- **Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`\n- **Shape manipulation**: `reshape()`, `flatten()`, `transpose()`\n- **Indexing**: Slicing, boolean indexing, fancy indexing\n\n### Essential Operations\n- **Element-wise**: Addition, multiplication, mathematical functions\n- **Aggregations**: `sum()`, `mean()`, `std()`, `min()`, `max()`\n- **Broadcasting**: Automatic expansion of arrays for operations\n\n### Why It Matters\nEvery ML algorithm processes data as arrays. Understanding these operations lets you:\n- Prepare data efficiently\n- Debug model issues\n- Optimize performance\n\nLet's practice with hands-on problems!\n    ",problems:["numpy-array-sum","numpy-matrix-multiply","numpy-broadcast-add"]},{id:"einsum",title:"Einstein Summation (Einsum)",description:"Master einsum for elegant tensor operations.",icon:"∑",introduction:`
# Einstein Summation (Einsum)

Einsum is a powerful notation for expressing tensor operations concisely.

## Why Einsum?
- Express complex operations in one line
- Often faster than chained operations
- Essential for attention mechanisms

## Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

## Common Patterns

### Basic Operations
\`\`\`
'ij->'      Sum all elements
'ij->i'     Sum along axis 1 (row sums)
'ij->j'     Sum along axis 0 (column sums)
'ij->ji'    Transpose
'ii->i'     Extract diagonal
'ii->'      Trace (sum of diagonal)
\`\`\`

### Matrix Operations
\`\`\`
'ik,kj->ij'   Matrix multiplication (A @ B)
'ij,ij->ij'   Element-wise product (A * B)
'ij,ij->'     Frobenius inner product
'i,j->ij'     Outer product
'i,i->'       Dot product
\`\`\`

### Batch Operations
\`\`\`
'bij,bjk->bik'   Batch matrix multiply
'bqd,bkd->bqk'   Attention scores (Q @ K.T)
'bhqk,bhkd->bhqd' Multi-head attention output
\`\`\`

## Key Insight
Letters that appear on both sides are kept.
Letters that disappear are summed over (contracted).

Master einsum to write clean, efficient deep learning code!
    `,problems:["einsum-basics","einsum-matrix-ops","einsum-batch-ops","einsum-advanced","einsum-vs-matmul"]},{id:"pytorch-basics",title:"PyTorch Basics",description:"Learn PyTorch patterns implemented in NumPy.",icon:"🔥",introduction:`
# PyTorch Basics

Learn PyTorch patterns and concepts. Since Pyodide doesn't support PyTorch, we implement these concepts in NumPy.

## Tensor Operations

### Creation
\`\`\`python
# PyTorch           # NumPy equivalent
torch.tensor()      np.array()
torch.zeros()       np.zeros()
torch.randn()       np.random.randn()
\`\`\`

### Operations
\`\`\`python
# PyTorch           # NumPy equivalent
x + y               x + y
x @ y               x @ y
x.sum(dim=1)        x.sum(axis=1)
x.view(2, -1)       x.reshape(2, -1)
\`\`\`

## Autograd Concepts

PyTorch tracks operations for automatic differentiation:
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # dy/dx = 2x = 4
\`\`\`

We implement this manually to understand the math.

## Module Pattern
\`\`\`python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def forward(self, x):
        return x @ self.weight + self.bias
\`\`\`

## Loss Functions
- CrossEntropyLoss: Classification
- MSELoss: Regression
- BCELoss: Binary classification

Learn these patterns to understand PyTorch!
    `,problems:["tensor-creation","tensor-operations","autograd-concepts","nn-modules","loss-functions"]},{id:"data-preprocessing",title:"Data Preprocessing",description:"Learn essential data cleaning and transformation techniques.",icon:"🧹",introduction:`
# Data Preprocessing

Real-world data is messy. Before training any model, you need to clean and transform your data.

## Key Concepts

### Handling Missing Data
- **Detection**: \`np.isnan()\`, \`pd.isnull()\`
- **Strategies**: Remove, fill with mean/median/mode, interpolate

### Feature Scaling
- **Normalization**: Scale to [0, 1] range using min-max scaling
- **Standardization**: Transform to zero mean, unit variance (z-score)
- **When to use**: Most algorithms need scaled features

### Encoding Categorical Data
- **One-Hot Encoding**: Convert categories to binary vectors
- **Label Encoding**: Map categories to integers
- **When to use**: Tree-based models handle label encoding; others need one-hot

### Why It Matters
Garbage in, garbage out. Proper preprocessing can:
- Improve model accuracy by 10-30%
- Prevent training failures
- Reduce overfitting

Let's practice these essential skills!
    `,problems:["normalize-features","handle-missing-data","one-hot-encode"]},{id:"supervised-learning",title:"Supervised Learning",description:"Implement core supervised learning algorithms from scratch.",icon:"📊",introduction:`
# Supervised Learning

Supervised learning is when you train a model on labeled data to make predictions.

## Key Concepts

### Linear Regression
- **Goal**: Predict continuous values
- **Formula**: y = wx + b
- **Training**: Minimize mean squared error (MSE)
- **Gradient descent**: Update weights iteratively

### Logistic Regression
- **Goal**: Binary classification
- **Sigmoid function**: Maps outputs to [0, 1] probabilities
- **Loss**: Binary cross-entropy
- **Decision boundary**: Linear separation of classes

### Decision Trees
- **Concept**: Recursive binary splits based on features
- **Splitting criteria**: Gini impurity, Information gain
- **Advantages**: Interpretable, handles non-linear relationships

### The Interview Perspective
Interviewers often ask you to:
- Implement gradient descent from scratch
- Explain the math behind these algorithms
- Discuss trade-offs between algorithms

Let's build these algorithms!
    `,problems:["linear-regression-gd","logistic-regression","logistic-regression-full","binary-cross-entropy","decision-tree-split"]},{id:"unsupervised-learning",title:"Unsupervised Learning",description:"Implement clustering and dimensionality reduction algorithms.",icon:"🔍",introduction:`
# Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

## Key Concepts

### K-Means Clustering
- **Goal**: Group data into K clusters
- **Algorithm**:
  1. Initialize K centroids randomly
  2. Assign points to nearest centroid
  3. Update centroids as cluster means
  4. Repeat until convergence

### Principal Component Analysis (PCA)
- **Goal**: Reduce dimensionality while preserving variance
- **Steps**:
  1. Center the data (subtract mean)
  2. Compute covariance matrix
  3. Find eigenvectors/eigenvalues
  4. Project onto top components

### Why These Matter
- **K-Means**: Customer segmentation, image compression
- **PCA**: Feature reduction, visualization, noise removal

### Interview Tips
- Be ready to implement K-means from scratch
- Explain variance explained by components
- Discuss limitations (K-means assumes spherical clusters)

Let's implement these algorithms!
    `,problems:["kmeans-clustering","pca-implementation"]},{id:"model-evaluation",title:"Model Evaluation",description:"Learn metrics and techniques to evaluate ML models properly.",icon:"📈",introduction:`
# Model Evaluation

Building a model is only half the job. You need to evaluate it properly.

## Key Concepts

### Classification Metrics
- **Accuracy**: Correct / Total (can be misleading!)
- **Precision**: TP / (TP + FP) - "Of positive predictions, how many correct?"
- **Recall**: TP / (TP + FN) - "Of actual positives, how many found?"
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root MSE (same units as target)
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Proportion of variance explained

### Cross-Validation
- **Why**: Single train/test split is unreliable
- **K-Fold**: Split data into K parts, rotate test set
- **Stratified**: Maintain class distribution in folds

### Interview Essentials
- Know when to use each metric
- Implement cross-validation from scratch
- Discuss precision-recall trade-offs

Let's practice evaluation!
    `,problems:["precision-recall-f1","cross-validation","confusion-matrix"]},{id:"deep-learning",title:"Deep Learning Basics",description:"Activation functions and fundamental building blocks.",icon:"⚡",introduction:`
# Deep Learning Basics

Core building blocks that appear in every neural network.

## Key Concepts

### Activation Functions
- **ReLU**: max(0, x) - Most common, avoids vanishing gradient
- **Sigmoid**: 1/(1+e^-x) - Output in [0,1], used for binary classification
- **Softmax**: Converts logits to probabilities that sum to 1
- **Tanh**: Output in [-1, 1], zero-centered

### When to Use Each
- **Hidden layers**: ReLU (or variants like LeakyReLU)
- **Binary output**: Sigmoid
- **Multi-class output**: Softmax
- **RNNs**: Tanh (historically)

### Dense Layers
- **Forward pass**: output = activation(W @ x + b)
- **Parameters**: Weights W and biases b
- **Computation**: Matrix multiplication + bias + activation

Let's implement these fundamentals!
    `,problems:["perceptron","neural-network-forward","backpropagation"]},{id:"neural-networks",title:"Neural Networks",description:"Build neural networks from scratch with forward and backward passes.",icon:"🧠",introduction:`
# Neural Networks

Neural networks are the foundation of deep learning. Understanding the math behind them is essential for ML interviews.

## Key Concepts

### Multi-Layer Perceptron (MLP)
- **Architecture**: Input → Hidden layers → Output
- **Forward pass**: Compute activations layer by layer
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax

### Backpropagation
- **Chain rule**: Compute gradients layer by layer
- **Weight updates**: Gradient descent on all parameters
- **Key insight**: Errors propagate backwards through the network

### Training Techniques
- **Weight Initialization**: Xavier, He initialization
- **Batch Normalization**: Normalize activations for stable training
- **Dropout**: Randomly drop neurons to prevent overfitting

### Loss Functions
- **Cross-entropy**: For classification
- **MSE**: For regression

### Common Interview Topics
- Implement forward/backward pass from scratch
- Explain vanishing/exploding gradients
- Discuss initialization strategies
- Implement regularization techniques

Let's build neural networks from scratch!
    `,problems:["cross-entropy-loss","mlp-forward-backward","weight-init","batch-norm","dropout"]},{id:"cnn",title:"Convolutional Neural Networks",description:"Understand convolutions, pooling, and CNN architectures.",icon:"🖼️",introduction:`
# Convolutional Neural Networks (CNNs)

CNNs are the backbone of computer vision. They learn hierarchical features from images.

## Key Concepts

### Convolution Operation
- **Kernel/Filter**: Small matrix that slides over input
- **Feature maps**: Output of applying filters
- **Parameters**: Stride, padding, kernel size

### Pooling Layers
- **Max pooling**: Take maximum in each region
- **Average pooling**: Take average in each region
- **Purpose**: Reduce spatial dimensions, add translation invariance

### CNN Architecture Components
1. **Convolutional layers**: Extract features
2. **Pooling layers**: Reduce dimensions
3. **Flatten**: Convert 2D to 1D
4. **Fully connected**: Classification

### Output Size Formula
\`\`\`
output = (input - kernel + 2*padding) / stride + 1
\`\`\`

### Classic Architectures
- **LeNet**: First successful CNN
- **AlexNet**: Deep CNNs for ImageNet
- **VGG**: Very deep, small kernels
- **ResNet**: Skip connections

Let's implement CNN operations!
    `,problems:["conv-output-size","conv2d-forward","max-pool","flatten-layer","conv2d-advanced"]},{id:"transformers",title:"Attention & Transformers",description:"Master attention mechanisms and transformer architecture.",icon:"🤖",introduction:`
# Attention and Transformers

Transformers have revolutionized NLP and are now used in vision, speech, and more.

## Key Concepts

### Self-Attention
- **Query, Key, Value**: Three projections of input
- **Attention scores**: Q @ K.T / sqrt(d_k)
- **Output**: Weighted sum of values

### Scaled Dot-Product Attention
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

### Multi-Head Attention
- **Multiple heads**: Parallel attention with different projections
- **Concatenate**: Combine all head outputs
- **Project**: Linear transformation back to model dimension

### Transformer Components
- **Positional Encoding**: Inject position information
- **Layer Normalization**: Normalize across features
- **Feed-Forward Network**: MLP after attention
- **Residual Connections**: Add input to output

### Causal Masking
- **Purpose**: Prevent attending to future tokens
- **Implementation**: Add -inf to upper triangle before softmax

### Interview Essentials
- Implement scaled dot-product attention
- Explain why we scale by sqrt(d_k)
- Multi-head attention benefits
- Positional encoding purpose

Let's build transformers from scratch!
    `,problems:["scaled-dot-product-attention","multi-head-attention","positional-encoding","layer-norm","causal-mask"]},{id:"generative-models",title:"Generative Models",description:"Learn VAEs, diffusion models, and generative AI fundamentals.",icon:"🎨",introduction:`
# Generative Models

Generative models learn to create new data similar to the training distribution.

## Key Concepts

### Variational Autoencoders (VAEs)
- **Encoder**: Maps input to latent distribution (μ, σ)
- **Decoder**: Reconstructs input from latent sample
- **Reparameterization trick**: Enable backprop through sampling
- **Loss**: Reconstruction + KL divergence

### VAE Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_reconstructed||² + KL(q(z|x) || p(z))
\`\`\`

### Diffusion Models
- **Forward process**: Gradually add noise to data
- **Reverse process**: Learn to denoise
- **Noise schedule**: β_t controls noise at each step

### Key Formulas
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### KL Divergence
- Measures difference between distributions
- KL(P || Q) ≥ 0, equals 0 iff P = Q
- Not symmetric

### Interview Topics
- Explain reparameterization trick
- VAE loss components
- Diffusion forward/reverse process
- Compare GANs vs VAEs vs Diffusion

Let's implement generative models!
    `,problems:["kl-divergence","vae-reparameterization","vae-loss","vqvae-quantization","diffusion-noise-schedule","diffusion-forward"]},{id:"e2e-implementations",title:"End-to-End Implementations",description:"Build complete ML models from scratch.",icon:"🏗️",introduction:`
# End-to-End Implementations

Put everything together to build complete models from scratch.

## Why Build From Scratch?

1. **Deep Understanding**: Know every component
2. **Interview Preparation**: Common coding questions
3. **Debugging Skills**: Understand what can go wrong
4. **Framework Independence**: Adapt to any library

## Models Covered

### 2-Layer MLP with Backprop
Complete neural network with:
- Forward pass through hidden layer
- Backpropagation of gradients
- Weight updates via gradient descent

### Transformer Encoder
Full encoder block with:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Feed-forward network
- Residual connections

### Variational Autoencoder (VAE)
Generative model with:
- Encoder → latent distribution
- Reparameterization trick
- Decoder → reconstruction
- ELBO loss (reconstruction + KL)

### Diffusion Model
DDPM-style diffusion with:
- Noise schedule (β, α, ᾱ)
- Forward diffusion process
- Reverse denoising process
- Training objective

### Convolutional Neural Network
Image classifier with:
- Conv2D layers
- Max pooling
- Flatten + FC layers
- Full forward pass

Build these models to truly understand deep learning!
    `,problems:["e2e-mlp","e2e-transformer","e2e-vae","e2e-vqvae","e2e-diffusion","e2e-cnn"]}];function d1({isOpen:e,onClose:t}){const[n,r]=z.useState(""),[i,a]=z.useState(""),[o,s]=z.useState(!1),[l,u]=z.useState(!1),[d,c]=z.useState("");if(!e)return null;const p=async y=>{y.preventDefault(),s(!0),c("");try{(await fetch("https://formspree.io/f/mpqqwpav",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n,feedback:i,_subject:`ML Coding Lab Feedback from ${n}`})})).ok?(u(!0),r(""),a("")):c("Failed to send feedback. Please try again.")}catch{c("Failed to send feedback. Please try again.")}finally{s(!1)}},f=()=>{u(!1),c(""),t()};return x.jsxs("div",{className:"fixed inset-0 z-50 flex items-center justify-center",children:[x.jsx("div",{className:"absolute inset-0 bg-black bg-opacity-50",onClick:f}),x.jsxs("div",{className:"relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6",children:[x.jsx("button",{onClick:f,className:"absolute top-4 right-4 text-gray-400 hover:text-gray-600",children:x.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:x.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M6 18L18 6M6 6l12 12"})})}),x.jsx("h2",{className:"text-xl font-semibold text-gray-900 mb-4",children:"Send Feedback"}),l?x.jsxs("div",{className:"text-center py-8",children:[x.jsx("div",{className:"w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4",children:x.jsx("svg",{className:"w-6 h-6 text-green-600",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:x.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M5 13l4 4L19 7"})})}),x.jsx("p",{className:"text-gray-700 mb-4",children:"Thank you for your feedback!"}),x.jsx("button",{onClick:f,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors",children:"Close"})]}):x.jsxs("form",{onSubmit:p,className:"space-y-4",children:[x.jsxs("div",{children:[x.jsx("label",{htmlFor:"name",className:"block text-sm font-medium text-gray-700 mb-1",children:"Name"}),x.jsx("input",{type:"text",id:"name",value:n,onChange:y=>r(y.target.value),required:!0,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent",placeholder:"Your name"})]}),x.jsxs("div",{children:[x.jsx("label",{htmlFor:"feedback",className:"block text-sm font-medium text-gray-700 mb-1",children:"Feedback"}),x.jsx("textarea",{id:"feedback",value:i,onChange:y=>a(y.target.value),required:!0,rows:4,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none",placeholder:"Share your thoughts, suggestions, or report issues..."})]}),d&&x.jsx("p",{className:"text-red-600 text-sm",children:d}),x.jsxs("div",{className:"flex gap-3 justify-end",children:[x.jsx("button",{type:"button",onClick:f,className:"px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors",children:"Cancel"}),x.jsx("button",{type:"submit",disabled:o,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:o?"Sending...":"Send Feedback"})]})]})]})]})}function p1(){const{getOverallProgress:e}=$i(),[t,n]=z.useState(!1),r=Nr.map(a=>({id:a.id,problemCount:a.problems.length})),i=e(r);return x.jsxs(x.Fragment,{children:[x.jsx("header",{className:"bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm",children:x.jsx("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",children:x.jsxs("div",{className:"flex items-center justify-between h-16",children:[x.jsxs(qt,{to:"/",className:"flex items-center gap-3",children:[x.jsx("div",{className:"w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center",children:x.jsx("span",{className:"text-white font-bold text-lg",children:"ML"})}),x.jsx("span",{className:"text-xl font-semibold text-gray-900",children:"ML Coding Lab"})]}),x.jsxs("div",{className:"flex items-center gap-6",children:[x.jsxs(qt,{to:"/scratchpad",className:"px-3 py-1.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-200 transition-colors flex items-center gap-2",children:[x.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:x.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"})}),"Scratchpad"]}),x.jsxs("a",{href:"/ml-coding-lab/ml-cheatsheet.html",target:"_blank",rel:"noopener noreferrer",className:"px-3 py-1.5 bg-amber-100 text-amber-700 text-sm font-medium rounded-md hover:bg-amber-200 transition-colors flex items-center gap-2",children:[x.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:x.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"})}),"Cheat Sheet"]}),x.jsxs("button",{onClick:()=>n(!0),className:"px-3 py-1.5 text-gray-500 text-sm font-medium hover:text-gray-700 transition-colors flex items-center gap-1",children:[x.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:x.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"})}),"Feedback"]}),x.jsxs("div",{className:"flex items-center gap-3",children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Overall Progress"}),x.jsx("div",{className:"w-32 h-2 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${i}%`}})}),x.jsxs("span",{className:"text-primary-600 text-sm font-medium",children:[i,"%"]})]})]})]})})}),x.jsx(d1,{isOpen:t,onClose:()=>n(!1)})]})}function f1(){const{getSectionProgress:e}=$i();return x.jsx("aside",{className:"w-64 bg-white border-r border-gray-200 min-h-[calc(100vh-4rem)] p-4",children:x.jsxs("nav",{className:"space-y-2",children:[x.jsx("h3",{className:"text-gray-500 text-xs font-semibold uppercase tracking-wider mb-4 px-3",children:"Learning Sections"}),Nr.map((t,n)=>{const r=e(t.id,t.problems.length);return x.jsxs(o1,{to:`/section/${t.id}`,className:({isActive:i})=>`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${i?"bg-primary-50 text-primary-700":"text-gray-700 hover:bg-gray-100 hover:text-gray-900"}`,children:[x.jsx("span",{className:"w-6 h-6 flex items-center justify-center text-lg",children:t.icon}),x.jsxs("div",{className:"flex-1 min-w-0",children:[x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsxs("span",{className:"text-gray-400 text-xs",children:[n+1,"."]}),x.jsx("span",{className:"text-sm font-medium truncate",children:t.title})]}),x.jsxs("div",{className:"flex items-center gap-2 mt-1",children:[x.jsx("div",{className:"flex-1 h-1 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${r}%`}})}),x.jsxs("span",{className:"text-xs text-gray-400",children:[r,"%"]})]})]})]},t.id)})]})})}function m1({children:e}){return x.jsxs("div",{className:"min-h-screen bg-gray-50",children:[x.jsx(p1,{}),x.jsxs("div",{className:"flex",children:[x.jsx(f1,{}),x.jsx("main",{className:"flex-1 p-6",children:e})]})]})}function h1(){const{getSectionProgress:e,getOverallProgress:t}=$i(),n=Nr.map(i=>({id:i.id,problemCount:i.problems.length})),r=t(n);return x.jsxs("div",{className:"max-w-5xl mx-auto",children:[x.jsxs("div",{className:"text-center mb-12",children:[x.jsx("h1",{className:"text-4xl font-bold text-gray-900 mb-4",children:"Learn ML By Building"}),x.jsx("p",{className:"text-lg text-gray-600 max-w-2xl mx-auto mb-8",children:"Practice hands-on machine learning coding problems. Run Python code directly in your browser with NumPy and scikit-learn support."}),x.jsxs("div",{className:"inline-flex items-center gap-4 bg-white rounded-lg px-6 py-3 shadow-sm border border-gray-200",children:[x.jsx("span",{className:"text-gray-600",children:"Overall Progress"}),x.jsx("div",{className:"w-48 h-2 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-500",style:{width:`${r}%`}})}),x.jsxs("span",{className:"text-primary-600 font-medium",children:[r,"%"]})]})]}),x.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:Nr.map((i,a)=>{const o=e(i.id,i.problems.length);return x.jsx(qt,{to:`/section/${i.id}`,className:"group bg-white rounded-xl p-6 border border-gray-200 hover:border-primary-400 hover:shadow-md transition-all duration-200",children:x.jsxs("div",{className:"flex items-start gap-4",children:[x.jsx("div",{className:"w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center text-2xl group-hover:bg-primary-50 transition-colors",children:i.icon}),x.jsxs("div",{className:"flex-1",children:[x.jsxs("div",{className:"flex items-center gap-2 mb-1",children:[x.jsxs("span",{className:"text-gray-400 text-sm",children:[a+1,"."]}),x.jsx("h3",{className:"text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors",children:i.title})]}),x.jsx("p",{className:"text-gray-500 text-sm mb-4",children:i.description}),x.jsxs("div",{className:"flex items-center justify-between",children:[x.jsxs("span",{className:"text-gray-400 text-sm",children:[i.problems.length," problems"]}),x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("div",{className:"w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${o}%`}})}),x.jsxs("span",{className:"text-xs text-gray-500",children:[o,"%"]})]})]})]})]})},i.id)})}),x.jsxs("div",{className:"mt-16 grid grid-cols-1 md:grid-cols-3 gap-6",children:[x.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[x.jsx("div",{className:"text-3xl mb-3",children:"🐍"}),x.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Python in Browser"}),x.jsx("p",{className:"text-gray-500 text-sm",children:"Run Python code with NumPy, pandas, and more - no setup required."})]}),x.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[x.jsx("div",{className:"text-3xl mb-3",children:"✅"}),x.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Instant Feedback"}),x.jsx("p",{className:"text-gray-500 text-sm",children:"Test your code against multiple test cases and see results immediately."})]}),x.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[x.jsx("div",{className:"text-3xl mb-3",children:"📊"}),x.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Track Progress"}),x.jsx("p",{className:"text-gray-500 text-sm",children:"Your progress is saved locally. Pick up where you left off anytime."})]})]})]})}function g1(e,t){const n={};return(e[e.length-1]===""?[...e,""]:e).join((n.padRight?" ":"")+","+(n.padLeft===!1?"":" ")).trim()}const y1=/^[$_\p{ID_Start}][$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,x1=/^[$_\p{ID_Start}][-$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,_1={};function Cd(e,t){return(_1.jsx?x1:y1).test(e)}const v1=/[ \t\n\f\r]/g;function w1(e){return typeof e=="object"?e.type==="text"?zd(e.value):!1:zd(e)}function zd(e){return e.replace(v1,"")===""}class Qi{constructor(t,n,r){this.normal=n,this.property=t,r&&(this.space=r)}}Qi.prototype.normal={};Qi.prototype.property={};Qi.prototype.space=void 0;function rh(e,t){const n={},r={};for(const i of e)Object.assign(n,i.property),Object.assign(r,i.normal);return new Qi(n,r,t)}function Rl(e){return e.toLowerCase()}class Je{constructor(t,n){this.attribute=n,this.property=t}}Je.prototype.attribute="";Je.prototype.booleanish=!1;Je.prototype.boolean=!1;Je.prototype.commaOrSpaceSeparated=!1;Je.prototype.commaSeparated=!1;Je.prototype.defined=!1;Je.prototype.mustUseProperty=!1;Je.prototype.number=!1;Je.prototype.overloadedBoolean=!1;Je.prototype.property="";Je.prototype.spaceSeparated=!1;Je.prototype.space=void 0;let k1=0;const G=Yn(),Se=Yn(),jl=Yn(),R=Yn(),de=Yn(),vr=Yn(),tt=Yn();function Yn(){return 2**++k1}const Il=Object.freeze(Object.defineProperty({__proto__:null,boolean:G,booleanish:Se,commaOrSpaceSeparated:tt,commaSeparated:vr,number:R,overloadedBoolean:jl,spaceSeparated:de},Symbol.toStringTag,{value:"Module"})),gs=Object.keys(Il);class qu extends Je{constructor(t,n,r,i){let a=-1;if(super(t,n),Ed(this,"space",i),typeof r=="number")for(;++a<gs.length;){const o=gs[a];Ed(this,gs[a],(r&Il[o])===Il[o])}}}qu.prototype.defined=!0;function Ed(e,t,n){n&&(e[t]=n)}function Ir(e){const t={},n={};for(const[r,i]of Object.entries(e.properties)){const a=new qu(r,e.transform(e.attributes||{},r),i,e.space);e.mustUseProperty&&e.mustUseProperty.includes(r)&&(a.mustUseProperty=!0),t[r]=a,n[Rl(r)]=r,n[Rl(a.attribute)]=r}return new Qi(t,n,e.space)}const ih=Ir({properties:{ariaActiveDescendant:null,ariaAtomic:Se,ariaAutoComplete:null,ariaBusy:Se,ariaChecked:Se,ariaColCount:R,ariaColIndex:R,ariaColSpan:R,ariaControls:de,ariaCurrent:null,ariaDescribedBy:de,ariaDetails:null,ariaDisabled:Se,ariaDropEffect:de,ariaErrorMessage:null,ariaExpanded:Se,ariaFlowTo:de,ariaGrabbed:Se,ariaHasPopup:null,ariaHidden:Se,ariaInvalid:null,ariaKeyShortcuts:null,ariaLabel:null,ariaLabelledBy:de,ariaLevel:R,ariaLive:null,ariaModal:Se,ariaMultiLine:Se,ariaMultiSelectable:Se,ariaOrientation:null,ariaOwns:de,ariaPlaceholder:null,ariaPosInSet:R,ariaPressed:Se,ariaReadOnly:Se,ariaRelevant:null,ariaRequired:Se,ariaRoleDescription:de,ariaRowCount:R,ariaRowIndex:R,ariaRowSpan:R,ariaSelected:Se,ariaSetSize:R,ariaSort:null,ariaValueMax:R,ariaValueMin:R,ariaValueNow:R,ariaValueText:null,role:null},transform(e,t){return t==="role"?t:"aria-"+t.slice(4).toLowerCase()}});function ah(e,t){return t in e?e[t]:t}function oh(e,t){return ah(e,t.toLowerCase())}const b1=Ir({attributes:{acceptcharset:"accept-charset",classname:"class",htmlfor:"for",httpequiv:"http-equiv"},mustUseProperty:["checked","multiple","muted","selected"],properties:{abbr:null,accept:vr,acceptCharset:de,accessKey:de,action:null,allow:null,allowFullScreen:G,allowPaymentRequest:G,allowUserMedia:G,alt:null,as:null,async:G,autoCapitalize:null,autoComplete:de,autoFocus:G,autoPlay:G,blocking:de,capture:null,charSet:null,checked:G,cite:null,className:de,cols:R,colSpan:null,content:null,contentEditable:Se,controls:G,controlsList:de,coords:R|vr,crossOrigin:null,data:null,dateTime:null,decoding:null,default:G,defer:G,dir:null,dirName:null,disabled:G,download:jl,draggable:Se,encType:null,enterKeyHint:null,fetchPriority:null,form:null,formAction:null,formEncType:null,formMethod:null,formNoValidate:G,formTarget:null,headers:de,height:R,hidden:jl,high:R,href:null,hrefLang:null,htmlFor:de,httpEquiv:de,id:null,imageSizes:null,imageSrcSet:null,inert:G,inputMode:null,integrity:null,is:null,isMap:G,itemId:null,itemProp:de,itemRef:de,itemScope:G,itemType:de,kind:null,label:null,lang:null,language:null,list:null,loading:null,loop:G,low:R,manifest:null,max:null,maxLength:R,media:null,method:null,min:null,minLength:R,multiple:G,muted:G,name:null,nonce:null,noModule:G,noValidate:G,onAbort:null,onAfterPrint:null,onAuxClick:null,onBeforeMatch:null,onBeforePrint:null,onBeforeToggle:null,onBeforeUnload:null,onBlur:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onContextLost:null,onContextMenu:null,onContextRestored:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnded:null,onError:null,onFocus:null,onFormData:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLanguageChange:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadEnd:null,onLoadStart:null,onMessage:null,onMessageError:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRejectionHandled:null,onReset:null,onResize:null,onScroll:null,onScrollEnd:null,onSecurityPolicyViolation:null,onSeeked:null,onSeeking:null,onSelect:null,onSlotChange:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnhandledRejection:null,onUnload:null,onVolumeChange:null,onWaiting:null,onWheel:null,open:G,optimum:R,pattern:null,ping:de,placeholder:null,playsInline:G,popover:null,popoverTarget:null,popoverTargetAction:null,poster:null,preload:null,readOnly:G,referrerPolicy:null,rel:de,required:G,reversed:G,rows:R,rowSpan:R,sandbox:de,scope:null,scoped:G,seamless:G,selected:G,shadowRootClonable:G,shadowRootDelegatesFocus:G,shadowRootMode:null,shape:null,size:R,sizes:null,slot:null,span:R,spellCheck:Se,src:null,srcDoc:null,srcLang:null,srcSet:null,start:R,step:null,style:null,tabIndex:R,target:null,title:null,translate:null,type:null,typeMustMatch:G,useMap:null,value:Se,width:R,wrap:null,writingSuggestions:null,align:null,aLink:null,archive:de,axis:null,background:null,bgColor:null,border:R,borderColor:null,bottomMargin:R,cellPadding:null,cellSpacing:null,char:null,charOff:null,classId:null,clear:null,code:null,codeBase:null,codeType:null,color:null,compact:G,declare:G,event:null,face:null,frame:null,frameBorder:null,hSpace:R,leftMargin:R,link:null,longDesc:null,lowSrc:null,marginHeight:R,marginWidth:R,noResize:G,noHref:G,noShade:G,noWrap:G,object:null,profile:null,prompt:null,rev:null,rightMargin:R,rules:null,scheme:null,scrolling:Se,standby:null,summary:null,text:null,topMargin:R,valueType:null,version:null,vAlign:null,vLink:null,vSpace:R,allowTransparency:null,autoCorrect:null,autoSave:null,disablePictureInPicture:G,disableRemotePlayback:G,prefix:null,property:null,results:R,security:null,unselectable:null},space:"html",transform:oh}),S1=Ir({attributes:{accentHeight:"accent-height",alignmentBaseline:"alignment-baseline",arabicForm:"arabic-form",baselineShift:"baseline-shift",capHeight:"cap-height",className:"class",clipPath:"clip-path",clipRule:"clip-rule",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",crossOrigin:"crossorigin",dataType:"datatype",dominantBaseline:"dominant-baseline",enableBackground:"enable-background",fillOpacity:"fill-opacity",fillRule:"fill-rule",floodColor:"flood-color",floodOpacity:"flood-opacity",fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",hrefLang:"hreflang",horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",horizOriginY:"horiz-origin-y",imageRendering:"image-rendering",letterSpacing:"letter-spacing",lightingColor:"lighting-color",markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",navDown:"nav-down",navDownLeft:"nav-down-left",navDownRight:"nav-down-right",navLeft:"nav-left",navNext:"nav-next",navPrev:"nav-prev",navRight:"nav-right",navUp:"nav-up",navUpLeft:"nav-up-left",navUpRight:"nav-up-right",onAbort:"onabort",onActivate:"onactivate",onAfterPrint:"onafterprint",onBeforePrint:"onbeforeprint",onBegin:"onbegin",onCancel:"oncancel",onCanPlay:"oncanplay",onCanPlayThrough:"oncanplaythrough",onChange:"onchange",onClick:"onclick",onClose:"onclose",onCopy:"oncopy",onCueChange:"oncuechange",onCut:"oncut",onDblClick:"ondblclick",onDrag:"ondrag",onDragEnd:"ondragend",onDragEnter:"ondragenter",onDragExit:"ondragexit",onDragLeave:"ondragleave",onDragOver:"ondragover",onDragStart:"ondragstart",onDrop:"ondrop",onDurationChange:"ondurationchange",onEmptied:"onemptied",onEnd:"onend",onEnded:"onended",onError:"onerror",onFocus:"onfocus",onFocusIn:"onfocusin",onFocusOut:"onfocusout",onHashChange:"onhashchange",onInput:"oninput",onInvalid:"oninvalid",onKeyDown:"onkeydown",onKeyPress:"onkeypress",onKeyUp:"onkeyup",onLoad:"onload",onLoadedData:"onloadeddata",onLoadedMetadata:"onloadedmetadata",onLoadStart:"onloadstart",onMessage:"onmessage",onMouseDown:"onmousedown",onMouseEnter:"onmouseenter",onMouseLeave:"onmouseleave",onMouseMove:"onmousemove",onMouseOut:"onmouseout",onMouseOver:"onmouseover",onMouseUp:"onmouseup",onMouseWheel:"onmousewheel",onOffline:"onoffline",onOnline:"ononline",onPageHide:"onpagehide",onPageShow:"onpageshow",onPaste:"onpaste",onPause:"onpause",onPlay:"onplay",onPlaying:"onplaying",onPopState:"onpopstate",onProgress:"onprogress",onRateChange:"onratechange",onRepeat:"onrepeat",onReset:"onreset",onResize:"onresize",onScroll:"onscroll",onSeeked:"onseeked",onSeeking:"onseeking",onSelect:"onselect",onShow:"onshow",onStalled:"onstalled",onStorage:"onstorage",onSubmit:"onsubmit",onSuspend:"onsuspend",onTimeUpdate:"ontimeupdate",onToggle:"ontoggle",onUnload:"onunload",onVolumeChange:"onvolumechange",onWaiting:"onwaiting",onZoom:"onzoom",overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pointerEvents:"pointer-events",referrerPolicy:"referrerpolicy",renderingIntent:"rendering-intent",shapeRendering:"shape-rendering",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",strokeDashArray:"stroke-dasharray",strokeDashOffset:"stroke-dashoffset",strokeLineCap:"stroke-linecap",strokeLineJoin:"stroke-linejoin",strokeMiterLimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",tabIndex:"tabindex",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",transformOrigin:"transform-origin",typeOf:"typeof",underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",vectorEffect:"vector-effect",vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",wordSpacing:"word-spacing",writingMode:"writing-mode",xHeight:"x-height",playbackOrder:"playbackorder",timelineBegin:"timelinebegin"},properties:{about:tt,accentHeight:R,accumulate:null,additive:null,alignmentBaseline:null,alphabetic:R,amplitude:R,arabicForm:null,ascent:R,attributeName:null,attributeType:null,azimuth:R,bandwidth:null,baselineShift:null,baseFrequency:null,baseProfile:null,bbox:null,begin:null,bias:R,by:null,calcMode:null,capHeight:R,className:de,clip:null,clipPath:null,clipPathUnits:null,clipRule:null,color:null,colorInterpolation:null,colorInterpolationFilters:null,colorProfile:null,colorRendering:null,content:null,contentScriptType:null,contentStyleType:null,crossOrigin:null,cursor:null,cx:null,cy:null,d:null,dataType:null,defaultAction:null,descent:R,diffuseConstant:R,direction:null,display:null,dur:null,divisor:R,dominantBaseline:null,download:G,dx:null,dy:null,edgeMode:null,editable:null,elevation:R,enableBackground:null,end:null,event:null,exponent:R,externalResourcesRequired:null,fill:null,fillOpacity:R,fillRule:null,filter:null,filterRes:null,filterUnits:null,floodColor:null,floodOpacity:null,focusable:null,focusHighlight:null,fontFamily:null,fontSize:null,fontSizeAdjust:null,fontStretch:null,fontStyle:null,fontVariant:null,fontWeight:null,format:null,fr:null,from:null,fx:null,fy:null,g1:vr,g2:vr,glyphName:vr,glyphOrientationHorizontal:null,glyphOrientationVertical:null,glyphRef:null,gradientTransform:null,gradientUnits:null,handler:null,hanging:R,hatchContentUnits:null,hatchUnits:null,height:null,href:null,hrefLang:null,horizAdvX:R,horizOriginX:R,horizOriginY:R,id:null,ideographic:R,imageRendering:null,initialVisibility:null,in:null,in2:null,intercept:R,k:R,k1:R,k2:R,k3:R,k4:R,kernelMatrix:tt,kernelUnitLength:null,keyPoints:null,keySplines:null,keyTimes:null,kerning:null,lang:null,lengthAdjust:null,letterSpacing:null,lightingColor:null,limitingConeAngle:R,local:null,markerEnd:null,markerMid:null,markerStart:null,markerHeight:null,markerUnits:null,markerWidth:null,mask:null,maskContentUnits:null,maskUnits:null,mathematical:null,max:null,media:null,mediaCharacterEncoding:null,mediaContentEncodings:null,mediaSize:R,mediaTime:null,method:null,min:null,mode:null,name:null,navDown:null,navDownLeft:null,navDownRight:null,navLeft:null,navNext:null,navPrev:null,navRight:null,navUp:null,navUpLeft:null,navUpRight:null,numOctaves:null,observer:null,offset:null,onAbort:null,onActivate:null,onAfterPrint:null,onBeforePrint:null,onBegin:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnd:null,onEnded:null,onError:null,onFocus:null,onFocusIn:null,onFocusOut:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadStart:null,onMessage:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onMouseWheel:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRepeat:null,onReset:null,onResize:null,onScroll:null,onSeeked:null,onSeeking:null,onSelect:null,onShow:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnload:null,onVolumeChange:null,onWaiting:null,onZoom:null,opacity:null,operator:null,order:null,orient:null,orientation:null,origin:null,overflow:null,overlay:null,overlinePosition:R,overlineThickness:R,paintOrder:null,panose1:null,path:null,pathLength:R,patternContentUnits:null,patternTransform:null,patternUnits:null,phase:null,ping:de,pitch:null,playbackOrder:null,pointerEvents:null,points:null,pointsAtX:R,pointsAtY:R,pointsAtZ:R,preserveAlpha:null,preserveAspectRatio:null,primitiveUnits:null,propagate:null,property:tt,r:null,radius:null,referrerPolicy:null,refX:null,refY:null,rel:tt,rev:tt,renderingIntent:null,repeatCount:null,repeatDur:null,requiredExtensions:tt,requiredFeatures:tt,requiredFonts:tt,requiredFormats:tt,resource:null,restart:null,result:null,rotate:null,rx:null,ry:null,scale:null,seed:null,shapeRendering:null,side:null,slope:null,snapshotTime:null,specularConstant:R,specularExponent:R,spreadMethod:null,spacing:null,startOffset:null,stdDeviation:null,stemh:null,stemv:null,stitchTiles:null,stopColor:null,stopOpacity:null,strikethroughPosition:R,strikethroughThickness:R,string:null,stroke:null,strokeDashArray:tt,strokeDashOffset:null,strokeLineCap:null,strokeLineJoin:null,strokeMiterLimit:R,strokeOpacity:R,strokeWidth:null,style:null,surfaceScale:R,syncBehavior:null,syncBehaviorDefault:null,syncMaster:null,syncTolerance:null,syncToleranceDefault:null,systemLanguage:tt,tabIndex:R,tableValues:null,target:null,targetX:R,targetY:R,textAnchor:null,textDecoration:null,textRendering:null,textLength:null,timelineBegin:null,title:null,transformBehavior:null,type:null,typeOf:tt,to:null,transform:null,transformOrigin:null,u1:null,u2:null,underlinePosition:R,underlineThickness:R,unicode:null,unicodeBidi:null,unicodeRange:null,unitsPerEm:R,values:null,vAlphabetic:R,vMathematical:R,vectorEffect:null,vHanging:R,vIdeographic:R,version:null,vertAdvY:R,vertOriginX:R,vertOriginY:R,viewBox:null,viewTarget:null,visibility:null,width:null,widths:null,wordSpacing:null,writingMode:null,x:null,x1:null,x2:null,xChannelSelector:null,xHeight:R,y:null,y1:null,y2:null,yChannelSelector:null,z:null,zoomAndPan:null},space:"svg",transform:ah}),sh=Ir({properties:{xLinkActuate:null,xLinkArcRole:null,xLinkHref:null,xLinkRole:null,xLinkShow:null,xLinkTitle:null,xLinkType:null},space:"xlink",transform(e,t){return"xlink:"+t.slice(5).toLowerCase()}}),lh=Ir({attributes:{xmlnsxlink:"xmlns:xlink"},properties:{xmlnsXLink:null,xmlns:null},space:"xmlns",transform:oh}),uh=Ir({properties:{xmlBase:null,xmlLang:null,xmlSpace:null},space:"xml",transform(e,t){return"xml:"+t.slice(3).toLowerCase()}}),C1={classId:"classID",dataType:"datatype",itemId:"itemID",strokeDashArray:"strokeDasharray",strokeDashOffset:"strokeDashoffset",strokeLineCap:"strokeLinecap",strokeLineJoin:"strokeLinejoin",strokeMiterLimit:"strokeMiterlimit",typeOf:"typeof",xLinkActuate:"xlinkActuate",xLinkArcRole:"xlinkArcrole",xLinkHref:"xlinkHref",xLinkRole:"xlinkRole",xLinkShow:"xlinkShow",xLinkTitle:"xlinkTitle",xLinkType:"xlinkType",xmlnsXLink:"xmlnsXlink"},z1=/[A-Z]/g,Pd=/-[a-z]/g,E1=/^data[-\w.:]+$/i;function P1(e,t){const n=Rl(t);let r=t,i=Je;if(n in e.normal)return e.property[e.normal[n]];if(n.length>4&&n.slice(0,4)==="data"&&E1.test(t)){if(t.charAt(4)==="-"){const a=t.slice(5).replace(Pd,T1);r="data"+a.charAt(0).toUpperCase()+a.slice(1)}else{const a=t.slice(4);if(!Pd.test(a)){let o=a.replace(z1,N1);o.charAt(0)!=="-"&&(o="-"+o),t="data"+o}}i=qu}return new i(r,t)}function N1(e){return"-"+e.toLowerCase()}function T1(e){return e.charAt(1).toUpperCase()}const L1=rh([ih,b1,sh,lh,uh],"html"),Wu=rh([ih,S1,sh,lh,uh],"svg");function A1(e){return e.join(" ").trim()}var Bu={},Nd=/\/\*[^*]*\*+([^/*][^*]*\*+)*\//g,R1=/\n/g,j1=/^\s*/,I1=/^(\*?[-#/*\\\w]+(\[[0-9a-z_-]+\])?)\s*/,D1=/^:\s*/,F1=/^((?:'(?:\\'|.)*?'|"(?:\\"|.)*?"|\([^)]*?\)|[^};])+)/,M1=/^[;\s]*/,O1=/^\s+|\s+$/g,q1=`
`,Td="/",Ld="*",Dn="",W1="comment",B1="declaration";function U1(e,t){if(typeof e!="string")throw new TypeError("First argument must be a string");if(!e)return[];t=t||{};var n=1,r=1;function i(y){var _=y.match(R1);_&&(n+=_.length);var S=y.lastIndexOf(q1);r=~S?y.length-S:r+y.length}function a(){var y={line:n,column:r};return function(_){return _.position=new o(y),u(),_}}function o(y){this.start=y,this.end={line:n,column:r},this.source=t.source}o.prototype.content=e;function s(y){var _=new Error(t.source+":"+n+":"+r+": "+y);if(_.reason=y,_.filename=t.source,_.line=n,_.column=r,_.source=e,!t.silent)throw _}function l(y){var _=y.exec(e);if(_){var S=_[0];return i(S),e=e.slice(S.length),_}}function u(){l(j1)}function d(y){var _;for(y=y||[];_=c();)_!==!1&&y.push(_);return y}function c(){var y=a();if(!(Td!=e.charAt(0)||Ld!=e.charAt(1))){for(var _=2;Dn!=e.charAt(_)&&(Ld!=e.charAt(_)||Td!=e.charAt(_+1));)++_;if(_+=2,Dn===e.charAt(_-1))return s("End of comment missing");var S=e.slice(2,_-2);return r+=2,i(S),e=e.slice(_),r+=2,y({type:W1,comment:S})}}function p(){var y=a(),_=l(I1);if(_){if(c(),!l(D1))return s("property missing ':'");var S=l(F1),m=y({type:B1,property:Ad(_[0].replace(Nd,Dn)),value:S?Ad(S[0].replace(Nd,Dn)):Dn});return l(M1),m}}function f(){var y=[];d(y);for(var _;_=p();)_!==!1&&(y.push(_),d(y));return y}return u(),f()}function Ad(e){return e?e.replace(O1,Dn):Dn}var V1=U1,H1=Da&&Da.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(Bu,"__esModule",{value:!0});Bu.default=Q1;const $1=H1(V1);function Q1(e,t){let n=null;if(!e||typeof e!="string")return n;const r=(0,$1.default)(e),i=typeof t=="function";return r.forEach(a=>{if(a.type!=="declaration")return;const{property:o,value:s}=a;i?t(o,s,a):s&&(n=n||{},n[o]=s)}),n}var Io={};Object.defineProperty(Io,"__esModule",{value:!0});Io.camelCase=void 0;var X1=/^--[a-zA-Z0-9_-]+$/,K1=/-([a-z])/g,Y1=/^[^-]+$/,G1=/^-(webkit|moz|ms|o|khtml)-/,Z1=/^-(ms)-/,J1=function(e){return!e||Y1.test(e)||X1.test(e)},e_=function(e,t){return t.toUpperCase()},Rd=function(e,t){return"".concat(t,"-")},t_=function(e,t){return t===void 0&&(t={}),J1(e)?e:(e=e.toLowerCase(),t.reactCompat?e=e.replace(Z1,Rd):e=e.replace(G1,Rd),e.replace(K1,e_))};Io.camelCase=t_;var n_=Da&&Da.__importDefault||function(e){return e&&e.__esModule?e:{default:e}},r_=n_(Bu),i_=Io;function Dl(e,t){var n={};return!e||typeof e!="string"||(0,r_.default)(e,function(r,i){r&&i&&(n[(0,i_.camelCase)(r,t)]=i)}),n}Dl.default=Dl;var a_=Dl;const o_=ho(a_),ch=dh("end"),Uu=dh("start");function dh(e){return t;function t(n){const r=n&&n.position&&n.position[e]||{};if(typeof r.line=="number"&&r.line>0&&typeof r.column=="number"&&r.column>0)return{line:r.line,column:r.column,offset:typeof r.offset=="number"&&r.offset>-1?r.offset:void 0}}}function s_(e){const t=Uu(e),n=ch(e);if(t&&n)return{start:t,end:n}}function di(e){return!e||typeof e!="object"?"":"position"in e||"type"in e?jd(e.position):"start"in e||"end"in e?jd(e):"line"in e||"column"in e?Fl(e):""}function Fl(e){return Id(e&&e.line)+":"+Id(e&&e.column)}function jd(e){return Fl(e&&e.start)+"-"+Fl(e&&e.end)}function Id(e){return e&&typeof e=="number"?e:1}class qe extends Error{constructor(t,n,r){super(),typeof n=="string"&&(r=n,n=void 0);let i="",a={},o=!1;if(n&&("line"in n&&"column"in n?a={place:n}:"start"in n&&"end"in n?a={place:n}:"type"in n?a={ancestors:[n],place:n.position}:a={...n}),typeof t=="string"?i=t:!a.cause&&t&&(o=!0,i=t.message,a.cause=t),!a.ruleId&&!a.source&&typeof r=="string"){const l=r.indexOf(":");l===-1?a.ruleId=r:(a.source=r.slice(0,l),a.ruleId=r.slice(l+1))}if(!a.place&&a.ancestors&&a.ancestors){const l=a.ancestors[a.ancestors.length-1];l&&(a.place=l.position)}const s=a.place&&"start"in a.place?a.place.start:a.place;this.ancestors=a.ancestors||void 0,this.cause=a.cause||void 0,this.column=s?s.column:void 0,this.fatal=void 0,this.file="",this.message=i,this.line=s?s.line:void 0,this.name=di(a.place)||"1:1",this.place=a.place||void 0,this.reason=this.message,this.ruleId=a.ruleId||void 0,this.source=a.source||void 0,this.stack=o&&a.cause&&typeof a.cause.stack=="string"?a.cause.stack:"",this.actual=void 0,this.expected=void 0,this.note=void 0,this.url=void 0}}qe.prototype.file="";qe.prototype.name="";qe.prototype.reason="";qe.prototype.message="";qe.prototype.stack="";qe.prototype.column=void 0;qe.prototype.line=void 0;qe.prototype.ancestors=void 0;qe.prototype.cause=void 0;qe.prototype.fatal=void 0;qe.prototype.place=void 0;qe.prototype.ruleId=void 0;qe.prototype.source=void 0;const Vu={}.hasOwnProperty,l_=new Map,u_=/[A-Z]/g,c_=new Set(["table","tbody","thead","tfoot","tr"]),d_=new Set(["td","th"]),ph="https://github.com/syntax-tree/hast-util-to-jsx-runtime";function p_(e,t){if(!t||t.Fragment===void 0)throw new TypeError("Expected `Fragment` in options");const n=t.filePath||void 0;let r;if(t.development){if(typeof t.jsxDEV!="function")throw new TypeError("Expected `jsxDEV` in options when `development: true`");r=v_(n,t.jsxDEV)}else{if(typeof t.jsx!="function")throw new TypeError("Expected `jsx` in production options");if(typeof t.jsxs!="function")throw new TypeError("Expected `jsxs` in production options");r=__(n,t.jsx,t.jsxs)}const i={Fragment:t.Fragment,ancestors:[],components:t.components||{},create:r,elementAttributeNameCase:t.elementAttributeNameCase||"react",evaluater:t.createEvaluater?t.createEvaluater():void 0,filePath:n,ignoreInvalidStyle:t.ignoreInvalidStyle||!1,passKeys:t.passKeys!==!1,passNode:t.passNode||!1,schema:t.space==="svg"?Wu:L1,stylePropertyNameCase:t.stylePropertyNameCase||"dom",tableCellAlignToStyle:t.tableCellAlignToStyle!==!1},a=fh(i,e,void 0);return a&&typeof a!="string"?a:i.create(e,i.Fragment,{children:a||void 0},void 0)}function fh(e,t,n){if(t.type==="element")return f_(e,t,n);if(t.type==="mdxFlowExpression"||t.type==="mdxTextExpression")return m_(e,t);if(t.type==="mdxJsxFlowElement"||t.type==="mdxJsxTextElement")return g_(e,t,n);if(t.type==="mdxjsEsm")return h_(e,t);if(t.type==="root")return y_(e,t,n);if(t.type==="text")return x_(e,t)}function f_(e,t,n){const r=e.schema;let i=r;t.tagName.toLowerCase()==="svg"&&r.space==="html"&&(i=Wu,e.schema=i),e.ancestors.push(t);const a=hh(e,t.tagName,!1),o=w_(e,t);let s=$u(e,t);return c_.has(t.tagName)&&(s=s.filter(function(l){return typeof l=="string"?!w1(l):!0})),mh(e,o,a,t),Hu(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function m_(e,t){if(t.data&&t.data.estree&&e.evaluater){const r=t.data.estree.body[0];return r.type,e.evaluater.evaluateExpression(r.expression)}ji(e,t.position)}function h_(e,t){if(t.data&&t.data.estree&&e.evaluater)return e.evaluater.evaluateProgram(t.data.estree);ji(e,t.position)}function g_(e,t,n){const r=e.schema;let i=r;t.name==="svg"&&r.space==="html"&&(i=Wu,e.schema=i),e.ancestors.push(t);const a=t.name===null?e.Fragment:hh(e,t.name,!0),o=k_(e,t),s=$u(e,t);return mh(e,o,a,t),Hu(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function y_(e,t,n){const r={};return Hu(r,$u(e,t)),e.create(t,e.Fragment,r,n)}function x_(e,t){return t.value}function mh(e,t,n,r){typeof n!="string"&&n!==e.Fragment&&e.passNode&&(t.node=r)}function Hu(e,t){if(t.length>0){const n=t.length>1?t:t[0];n&&(e.children=n)}}function __(e,t,n){return r;function r(i,a,o,s){const u=Array.isArray(o.children)?n:t;return s?u(a,o,s):u(a,o)}}function v_(e,t){return n;function n(r,i,a,o){const s=Array.isArray(a.children),l=Uu(r);return t(i,a,o,s,{columnNumber:l?l.column-1:void 0,fileName:e,lineNumber:l?l.line:void 0},void 0)}}function w_(e,t){const n={};let r,i;for(i in t.properties)if(i!=="children"&&Vu.call(t.properties,i)){const a=b_(e,i,t.properties[i]);if(a){const[o,s]=a;e.tableCellAlignToStyle&&o==="align"&&typeof s=="string"&&d_.has(t.tagName)?r=s:n[o]=s}}if(r){const a=n.style||(n.style={});a[e.stylePropertyNameCase==="css"?"text-align":"textAlign"]=r}return n}function k_(e,t){const n={};for(const r of t.attributes)if(r.type==="mdxJsxExpressionAttribute")if(r.data&&r.data.estree&&e.evaluater){const a=r.data.estree.body[0];a.type;const o=a.expression;o.type;const s=o.properties[0];s.type,Object.assign(n,e.evaluater.evaluateExpression(s.argument))}else ji(e,t.position);else{const i=r.name;let a;if(r.value&&typeof r.value=="object")if(r.value.data&&r.value.data.estree&&e.evaluater){const s=r.value.data.estree.body[0];s.type,a=e.evaluater.evaluateExpression(s.expression)}else ji(e,t.position);else a=r.value===null?!0:r.value;n[i]=a}return n}function $u(e,t){const n=[];let r=-1;const i=e.passKeys?new Map:l_;for(;++r<t.children.length;){const a=t.children[r];let o;if(e.passKeys){const l=a.type==="element"?a.tagName:a.type==="mdxJsxFlowElement"||a.type==="mdxJsxTextElement"?a.name:void 0;if(l){const u=i.get(l)||0;o=l+"-"+u,i.set(l,u+1)}}const s=fh(e,a,o);s!==void 0&&n.push(s)}return n}function b_(e,t,n){const r=P1(e.schema,t);if(!(n==null||typeof n=="number"&&Number.isNaN(n))){if(Array.isArray(n)&&(n=r.commaSeparated?g1(n):A1(n)),r.property==="style"){let i=typeof n=="object"?n:S_(e,String(n));return e.stylePropertyNameCase==="css"&&(i=C_(i)),["style",i]}return[e.elementAttributeNameCase==="react"&&r.space?C1[r.property]||r.property:r.attribute,n]}}function S_(e,t){try{return o_(t,{reactCompat:!0})}catch(n){if(e.ignoreInvalidStyle)return{};const r=n,i=new qe("Cannot parse `style` attribute",{ancestors:e.ancestors,cause:r,ruleId:"style",source:"hast-util-to-jsx-runtime"});throw i.file=e.filePath||void 0,i.url=ph+"#cannot-parse-style-attribute",i}}function hh(e,t,n){let r;if(!n)r={type:"Literal",value:t};else if(t.includes(".")){const i=t.split(".");let a=-1,o;for(;++a<i.length;){const s=Cd(i[a])?{type:"Identifier",name:i[a]}:{type:"Literal",value:i[a]};o=o?{type:"MemberExpression",object:o,property:s,computed:!!(a&&s.type==="Literal"),optional:!1}:s}r=o}else r=Cd(t)&&!/^[a-z]/.test(t)?{type:"Identifier",name:t}:{type:"Literal",value:t};if(r.type==="Literal"){const i=r.value;return Vu.call(e.components,i)?e.components[i]:i}if(e.evaluater)return e.evaluater.evaluateExpression(r);ji(e)}function ji(e,t){const n=new qe("Cannot handle MDX estrees without `createEvaluater`",{ancestors:e.ancestors,place:t,ruleId:"mdx-estree",source:"hast-util-to-jsx-runtime"});throw n.file=e.filePath||void 0,n.url=ph+"#cannot-handle-mdx-estrees-without-createevaluater",n}function C_(e){const t={};let n;for(n in e)Vu.call(e,n)&&(t[z_(n)]=e[n]);return t}function z_(e){let t=e.replace(u_,E_);return t.slice(0,3)==="ms-"&&(t="-"+t),t}function E_(e){return"-"+e.toLowerCase()}const ys={action:["form"],cite:["blockquote","del","ins","q"],data:["object"],formAction:["button","input"],href:["a","area","base","link"],icon:["menuitem"],itemId:null,manifest:["html"],ping:["a","area"],poster:["video"],src:["audio","embed","iframe","img","input","script","source","track","video"]},P_={};function Qu(e,t){const n=P_,r=typeof n.includeImageAlt=="boolean"?n.includeImageAlt:!0,i=typeof n.includeHtml=="boolean"?n.includeHtml:!0;return gh(e,r,i)}function gh(e,t,n){if(N_(e)){if("value"in e)return e.type==="html"&&!n?"":e.value;if(t&&"alt"in e&&e.alt)return e.alt;if("children"in e)return Dd(e.children,t,n)}return Array.isArray(e)?Dd(e,t,n):""}function Dd(e,t,n){const r=[];let i=-1;for(;++i<e.length;)r[i]=gh(e[i],t,n);return r.join("")}function N_(e){return!!(e&&typeof e=="object")}const Fd=document.createElement("i");function Xu(e){const t="&"+e+";";Fd.innerHTML=t;const n=Fd.textContent;return n.charCodeAt(n.length-1)===59&&e!=="semi"||n===t?!1:n}function ot(e,t,n,r){const i=e.length;let a=0,o;if(t<0?t=-t>i?0:i+t:t=t>i?i:t,n=n>0?n:0,r.length<1e4)o=Array.from(r),o.unshift(t,n),e.splice(...o);else for(n&&e.splice(t,n);a<r.length;)o=r.slice(a,a+1e4),o.unshift(t,0),e.splice(...o),a+=1e4,t+=1e4}function mt(e,t){return e.length>0?(ot(e,e.length,0,t),e):t}const Md={}.hasOwnProperty;function yh(e){const t={};let n=-1;for(;++n<e.length;)T_(t,e[n]);return t}function T_(e,t){let n;for(n in t){const i=(Md.call(e,n)?e[n]:void 0)||(e[n]={}),a=t[n];let o;if(a)for(o in a){Md.call(i,o)||(i[o]=[]);const s=a[o];L_(i[o],Array.isArray(s)?s:s?[s]:[])}}}function L_(e,t){let n=-1;const r=[];for(;++n<t.length;)(t[n].add==="after"?e:r).push(t[n]);ot(e,0,0,r)}function xh(e,t){const n=Number.parseInt(e,t);return n<9||n===11||n>13&&n<32||n>126&&n<160||n>55295&&n<57344||n>64975&&n<65008||(n&65535)===65535||(n&65535)===65534||n>1114111?"�":String.fromCodePoint(n)}function Tt(e){return e.replace(/[\t\n\r ]+/g," ").replace(/^ | $/g,"").toLowerCase().toUpperCase()}const Ue=Nn(/[A-Za-z]/),Me=Nn(/[\dA-Za-z]/),A_=Nn(/[#-'*+\--9=?A-Z^-~]/);function uo(e){return e!==null&&(e<32||e===127)}const Ml=Nn(/\d/),R_=Nn(/[\dA-Fa-f]/),j_=Nn(/[!-/:-@[-`{-~]/);function Q(e){return e!==null&&e<-2}function ce(e){return e!==null&&(e<0||e===32)}function Z(e){return e===-2||e===-1||e===32}const Do=Nn(new RegExp("\\p{P}|\\p{S}","u")),Qn=Nn(/\s/);function Nn(e){return t;function t(n){return n!==null&&n>-1&&e.test(String.fromCharCode(n))}}function Dr(e){const t=[];let n=-1,r=0,i=0;for(;++n<e.length;){const a=e.charCodeAt(n);let o="";if(a===37&&Me(e.charCodeAt(n+1))&&Me(e.charCodeAt(n+2)))i=2;else if(a<128)/[!#$&-;=?-Z_a-z~]/.test(String.fromCharCode(a))||(o=String.fromCharCode(a));else if(a>55295&&a<57344){const s=e.charCodeAt(n+1);a<56320&&s>56319&&s<57344?(o=String.fromCharCode(a,s),i=1):o="�"}else o=String.fromCharCode(a);o&&(t.push(e.slice(r,n),encodeURIComponent(o)),r=n+i+1,o=""),i&&(n+=i,i=0)}return t.join("")+e.slice(r)}function ne(e,t,n,r){const i=r?r-1:Number.POSITIVE_INFINITY;let a=0;return o;function o(l){return Z(l)?(e.enter(n),s(l)):t(l)}function s(l){return Z(l)&&a++<i?(e.consume(l),s):(e.exit(n),t(l))}}const I_={tokenize:D_};function D_(e){const t=e.attempt(this.parser.constructs.contentInitial,r,i);let n;return t;function r(s){if(s===null){e.consume(s);return}return e.enter("lineEnding"),e.consume(s),e.exit("lineEnding"),ne(e,t,"linePrefix")}function i(s){return e.enter("paragraph"),a(s)}function a(s){const l=e.enter("chunkText",{contentType:"text",previous:n});return n&&(n.next=l),n=l,o(s)}function o(s){if(s===null){e.exit("chunkText"),e.exit("paragraph"),e.consume(s);return}return Q(s)?(e.consume(s),e.exit("chunkText"),a):(e.consume(s),o)}}const F_={tokenize:M_},Od={tokenize:O_};function M_(e){const t=this,n=[];let r=0,i,a,o;return s;function s(g){if(r<n.length){const C=n[r];return t.containerState=C[1],e.attempt(C[0].continuation,l,u)(g)}return u(g)}function l(g){if(r++,t.containerState._closeFlow){t.containerState._closeFlow=void 0,i&&h();const C=t.events.length;let E=C,b;for(;E--;)if(t.events[E][0]==="exit"&&t.events[E][1].type==="chunkFlow"){b=t.events[E][1].end;break}m(r);let P=C;for(;P<t.events.length;)t.events[P][1].end={...b},P++;return ot(t.events,E+1,0,t.events.slice(C)),t.events.length=P,u(g)}return s(g)}function u(g){if(r===n.length){if(!i)return p(g);if(i.currentConstruct&&i.currentConstruct.concrete)return y(g);t.interrupt=!!(i.currentConstruct&&!i._gfmTableDynamicInterruptHack)}return t.containerState={},e.check(Od,d,c)(g)}function d(g){return i&&h(),m(r),p(g)}function c(g){return t.parser.lazy[t.now().line]=r!==n.length,o=t.now().offset,y(g)}function p(g){return t.containerState={},e.attempt(Od,f,y)(g)}function f(g){return r++,n.push([t.currentConstruct,t.containerState]),p(g)}function y(g){if(g===null){i&&h(),m(0),e.consume(g);return}return i=i||t.parser.flow(t.now()),e.enter("chunkFlow",{_tokenizer:i,contentType:"flow",previous:a}),_(g)}function _(g){if(g===null){S(e.exit("chunkFlow"),!0),m(0),e.consume(g);return}return Q(g)?(e.consume(g),S(e.exit("chunkFlow")),r=0,t.interrupt=void 0,s):(e.consume(g),_)}function S(g,C){const E=t.sliceStream(g);if(C&&E.push(null),g.previous=a,a&&(a.next=g),a=g,i.defineSkip(g.start),i.write(E),t.parser.lazy[g.start.line]){let b=i.events.length;for(;b--;)if(i.events[b][1].start.offset<o&&(!i.events[b][1].end||i.events[b][1].end.offset>o))return;const P=t.events.length;let T=P,D,k;for(;T--;)if(t.events[T][0]==="exit"&&t.events[T][1].type==="chunkFlow"){if(D){k=t.events[T][1].end;break}D=!0}for(m(r),b=P;b<t.events.length;)t.events[b][1].end={...k},b++;ot(t.events,T+1,0,t.events.slice(P)),t.events.length=b}}function m(g){let C=n.length;for(;C-- >g;){const E=n[C];t.containerState=E[1],E[0].exit.call(t,e)}n.length=g}function h(){i.write([null]),a=void 0,i=void 0,t.containerState._closeFlow=void 0}}function O_(e,t,n){return ne(e,e.attempt(this.parser.constructs.document,t,n),"linePrefix",this.parser.constructs.disable.null.includes("codeIndented")?void 0:4)}function Tr(e){if(e===null||ce(e)||Qn(e))return 1;if(Do(e))return 2}function Fo(e,t,n){const r=[];let i=-1;for(;++i<e.length;){const a=e[i].resolveAll;a&&!r.includes(a)&&(t=a(t,n),r.push(a))}return t}const Ol={name:"attention",resolveAll:q_,tokenize:W_};function q_(e,t){let n=-1,r,i,a,o,s,l,u,d;for(;++n<e.length;)if(e[n][0]==="enter"&&e[n][1].type==="attentionSequence"&&e[n][1]._close){for(r=n;r--;)if(e[r][0]==="exit"&&e[r][1].type==="attentionSequence"&&e[r][1]._open&&t.sliceSerialize(e[r][1]).charCodeAt(0)===t.sliceSerialize(e[n][1]).charCodeAt(0)){if((e[r][1]._close||e[n][1]._open)&&(e[n][1].end.offset-e[n][1].start.offset)%3&&!((e[r][1].end.offset-e[r][1].start.offset+e[n][1].end.offset-e[n][1].start.offset)%3))continue;l=e[r][1].end.offset-e[r][1].start.offset>1&&e[n][1].end.offset-e[n][1].start.offset>1?2:1;const c={...e[r][1].end},p={...e[n][1].start};qd(c,-l),qd(p,l),o={type:l>1?"strongSequence":"emphasisSequence",start:c,end:{...e[r][1].end}},s={type:l>1?"strongSequence":"emphasisSequence",start:{...e[n][1].start},end:p},a={type:l>1?"strongText":"emphasisText",start:{...e[r][1].end},end:{...e[n][1].start}},i={type:l>1?"strong":"emphasis",start:{...o.start},end:{...s.end}},e[r][1].end={...o.start},e[n][1].start={...s.end},u=[],e[r][1].end.offset-e[r][1].start.offset&&(u=mt(u,[["enter",e[r][1],t],["exit",e[r][1],t]])),u=mt(u,[["enter",i,t],["enter",o,t],["exit",o,t],["enter",a,t]]),u=mt(u,Fo(t.parser.constructs.insideSpan.null,e.slice(r+1,n),t)),u=mt(u,[["exit",a,t],["enter",s,t],["exit",s,t],["exit",i,t]]),e[n][1].end.offset-e[n][1].start.offset?(d=2,u=mt(u,[["enter",e[n][1],t],["exit",e[n][1],t]])):d=0,ot(e,r-1,n-r+3,u),n=r+u.length-d-2;break}}for(n=-1;++n<e.length;)e[n][1].type==="attentionSequence"&&(e[n][1].type="data");return e}function W_(e,t){const n=this.parser.constructs.attentionMarkers.null,r=this.previous,i=Tr(r);let a;return o;function o(l){return a=l,e.enter("attentionSequence"),s(l)}function s(l){if(l===a)return e.consume(l),s;const u=e.exit("attentionSequence"),d=Tr(l),c=!d||d===2&&i||n.includes(l),p=!i||i===2&&d||n.includes(r);return u._open=!!(a===42?c:c&&(i||!p)),u._close=!!(a===42?p:p&&(d||!c)),t(l)}}function qd(e,t){e.column+=t,e.offset+=t,e._bufferIndex+=t}const B_={name:"autolink",tokenize:U_};function U_(e,t,n){let r=0;return i;function i(f){return e.enter("autolink"),e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.enter("autolinkProtocol"),a}function a(f){return Ue(f)?(e.consume(f),o):f===64?n(f):u(f)}function o(f){return f===43||f===45||f===46||Me(f)?(r=1,s(f)):u(f)}function s(f){return f===58?(e.consume(f),r=0,l):(f===43||f===45||f===46||Me(f))&&r++<32?(e.consume(f),s):(r=0,u(f))}function l(f){return f===62?(e.exit("autolinkProtocol"),e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.exit("autolink"),t):f===null||f===32||f===60||uo(f)?n(f):(e.consume(f),l)}function u(f){return f===64?(e.consume(f),d):A_(f)?(e.consume(f),u):n(f)}function d(f){return Me(f)?c(f):n(f)}function c(f){return f===46?(e.consume(f),r=0,d):f===62?(e.exit("autolinkProtocol").type="autolinkEmail",e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.exit("autolink"),t):p(f)}function p(f){if((f===45||Me(f))&&r++<63){const y=f===45?p:c;return e.consume(f),y}return n(f)}}const Xi={partial:!0,tokenize:V_};function V_(e,t,n){return r;function r(a){return Z(a)?ne(e,i,"linePrefix")(a):i(a)}function i(a){return a===null||Q(a)?t(a):n(a)}}const _h={continuation:{tokenize:$_},exit:Q_,name:"blockQuote",tokenize:H_};function H_(e,t,n){const r=this;return i;function i(o){if(o===62){const s=r.containerState;return s.open||(e.enter("blockQuote",{_container:!0}),s.open=!0),e.enter("blockQuotePrefix"),e.enter("blockQuoteMarker"),e.consume(o),e.exit("blockQuoteMarker"),a}return n(o)}function a(o){return Z(o)?(e.enter("blockQuotePrefixWhitespace"),e.consume(o),e.exit("blockQuotePrefixWhitespace"),e.exit("blockQuotePrefix"),t):(e.exit("blockQuotePrefix"),t(o))}}function $_(e,t,n){const r=this;return i;function i(o){return Z(o)?ne(e,a,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(o):a(o)}function a(o){return e.attempt(_h,t,n)(o)}}function Q_(e){e.exit("blockQuote")}const vh={name:"characterEscape",tokenize:X_};function X_(e,t,n){return r;function r(a){return e.enter("characterEscape"),e.enter("escapeMarker"),e.consume(a),e.exit("escapeMarker"),i}function i(a){return j_(a)?(e.enter("characterEscapeValue"),e.consume(a),e.exit("characterEscapeValue"),e.exit("characterEscape"),t):n(a)}}const wh={name:"characterReference",tokenize:K_};function K_(e,t,n){const r=this;let i=0,a,o;return s;function s(c){return e.enter("characterReference"),e.enter("characterReferenceMarker"),e.consume(c),e.exit("characterReferenceMarker"),l}function l(c){return c===35?(e.enter("characterReferenceMarkerNumeric"),e.consume(c),e.exit("characterReferenceMarkerNumeric"),u):(e.enter("characterReferenceValue"),a=31,o=Me,d(c))}function u(c){return c===88||c===120?(e.enter("characterReferenceMarkerHexadecimal"),e.consume(c),e.exit("characterReferenceMarkerHexadecimal"),e.enter("characterReferenceValue"),a=6,o=R_,d):(e.enter("characterReferenceValue"),a=7,o=Ml,d(c))}function d(c){if(c===59&&i){const p=e.exit("characterReferenceValue");return o===Me&&!Xu(r.sliceSerialize(p))?n(c):(e.enter("characterReferenceMarker"),e.consume(c),e.exit("characterReferenceMarker"),e.exit("characterReference"),t)}return o(c)&&i++<a?(e.consume(c),d):n(c)}}const Wd={partial:!0,tokenize:G_},Bd={concrete:!0,name:"codeFenced",tokenize:Y_};function Y_(e,t,n){const r=this,i={partial:!0,tokenize:E};let a=0,o=0,s;return l;function l(b){return u(b)}function u(b){const P=r.events[r.events.length-1];return a=P&&P[1].type==="linePrefix"?P[2].sliceSerialize(P[1],!0).length:0,s=b,e.enter("codeFenced"),e.enter("codeFencedFence"),e.enter("codeFencedFenceSequence"),d(b)}function d(b){return b===s?(o++,e.consume(b),d):o<3?n(b):(e.exit("codeFencedFenceSequence"),Z(b)?ne(e,c,"whitespace")(b):c(b))}function c(b){return b===null||Q(b)?(e.exit("codeFencedFence"),r.interrupt?t(b):e.check(Wd,_,C)(b)):(e.enter("codeFencedFenceInfo"),e.enter("chunkString",{contentType:"string"}),p(b))}function p(b){return b===null||Q(b)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),c(b)):Z(b)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),ne(e,f,"whitespace")(b)):b===96&&b===s?n(b):(e.consume(b),p)}function f(b){return b===null||Q(b)?c(b):(e.enter("codeFencedFenceMeta"),e.enter("chunkString",{contentType:"string"}),y(b))}function y(b){return b===null||Q(b)?(e.exit("chunkString"),e.exit("codeFencedFenceMeta"),c(b)):b===96&&b===s?n(b):(e.consume(b),y)}function _(b){return e.attempt(i,C,S)(b)}function S(b){return e.enter("lineEnding"),e.consume(b),e.exit("lineEnding"),m}function m(b){return a>0&&Z(b)?ne(e,h,"linePrefix",a+1)(b):h(b)}function h(b){return b===null||Q(b)?e.check(Wd,_,C)(b):(e.enter("codeFlowValue"),g(b))}function g(b){return b===null||Q(b)?(e.exit("codeFlowValue"),h(b)):(e.consume(b),g)}function C(b){return e.exit("codeFenced"),t(b)}function E(b,P,T){let D=0;return k;function k(F){return b.enter("lineEnding"),b.consume(F),b.exit("lineEnding"),I}function I(F){return b.enter("codeFencedFence"),Z(F)?ne(b,W,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(F):W(F)}function W(F){return F===s?(b.enter("codeFencedFenceSequence"),K(F)):T(F)}function K(F){return F===s?(D++,b.consume(F),K):D>=o?(b.exit("codeFencedFenceSequence"),Z(F)?ne(b,te,"whitespace")(F):te(F)):T(F)}function te(F){return F===null||Q(F)?(b.exit("codeFencedFence"),P(F)):T(F)}}}function G_(e,t,n){const r=this;return i;function i(o){return o===null?n(o):(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}const xs={name:"codeIndented",tokenize:J_},Z_={partial:!0,tokenize:ev};function J_(e,t,n){const r=this;return i;function i(u){return e.enter("codeIndented"),ne(e,a,"linePrefix",5)(u)}function a(u){const d=r.events[r.events.length-1];return d&&d[1].type==="linePrefix"&&d[2].sliceSerialize(d[1],!0).length>=4?o(u):n(u)}function o(u){return u===null?l(u):Q(u)?e.attempt(Z_,o,l)(u):(e.enter("codeFlowValue"),s(u))}function s(u){return u===null||Q(u)?(e.exit("codeFlowValue"),o(u)):(e.consume(u),s)}function l(u){return e.exit("codeIndented"),t(u)}}function ev(e,t,n){const r=this;return i;function i(o){return r.parser.lazy[r.now().line]?n(o):Q(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),i):ne(e,a,"linePrefix",5)(o)}function a(o){const s=r.events[r.events.length-1];return s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):Q(o)?i(o):n(o)}}const tv={name:"codeText",previous:rv,resolve:nv,tokenize:iv};function nv(e){let t=e.length-4,n=3,r,i;if((e[n][1].type==="lineEnding"||e[n][1].type==="space")&&(e[t][1].type==="lineEnding"||e[t][1].type==="space")){for(r=n;++r<t;)if(e[r][1].type==="codeTextData"){e[n][1].type="codeTextPadding",e[t][1].type="codeTextPadding",n+=2,t-=2;break}}for(r=n-1,t++;++r<=t;)i===void 0?r!==t&&e[r][1].type!=="lineEnding"&&(i=r):(r===t||e[r][1].type==="lineEnding")&&(e[i][1].type="codeTextData",r!==i+2&&(e[i][1].end=e[r-1][1].end,e.splice(i+2,r-i-2),t-=r-i-2,r=i+2),i=void 0);return e}function rv(e){return e!==96||this.events[this.events.length-1][1].type==="characterEscape"}function iv(e,t,n){let r=0,i,a;return o;function o(c){return e.enter("codeText"),e.enter("codeTextSequence"),s(c)}function s(c){return c===96?(e.consume(c),r++,s):(e.exit("codeTextSequence"),l(c))}function l(c){return c===null?n(c):c===32?(e.enter("space"),e.consume(c),e.exit("space"),l):c===96?(a=e.enter("codeTextSequence"),i=0,d(c)):Q(c)?(e.enter("lineEnding"),e.consume(c),e.exit("lineEnding"),l):(e.enter("codeTextData"),u(c))}function u(c){return c===null||c===32||c===96||Q(c)?(e.exit("codeTextData"),l(c)):(e.consume(c),u)}function d(c){return c===96?(e.consume(c),i++,d):i===r?(e.exit("codeTextSequence"),e.exit("codeText"),t(c)):(a.type="codeTextData",u(c))}}class av{constructor(t){this.left=t?[...t]:[],this.right=[]}get(t){if(t<0||t>=this.left.length+this.right.length)throw new RangeError("Cannot access index `"+t+"` in a splice buffer of size `"+(this.left.length+this.right.length)+"`");return t<this.left.length?this.left[t]:this.right[this.right.length-t+this.left.length-1]}get length(){return this.left.length+this.right.length}shift(){return this.setCursor(0),this.right.pop()}slice(t,n){const r=n??Number.POSITIVE_INFINITY;return r<this.left.length?this.left.slice(t,r):t>this.left.length?this.right.slice(this.right.length-r+this.left.length,this.right.length-t+this.left.length).reverse():this.left.slice(t).concat(this.right.slice(this.right.length-r+this.left.length).reverse())}splice(t,n,r){const i=n||0;this.setCursor(Math.trunc(t));const a=this.right.splice(this.right.length-i,Number.POSITIVE_INFINITY);return r&&Qr(this.left,r),a.reverse()}pop(){return this.setCursor(Number.POSITIVE_INFINITY),this.left.pop()}push(t){this.setCursor(Number.POSITIVE_INFINITY),this.left.push(t)}pushMany(t){this.setCursor(Number.POSITIVE_INFINITY),Qr(this.left,t)}unshift(t){this.setCursor(0),this.right.push(t)}unshiftMany(t){this.setCursor(0),Qr(this.right,t.reverse())}setCursor(t){if(!(t===this.left.length||t>this.left.length&&this.right.length===0||t<0&&this.left.length===0))if(t<this.left.length){const n=this.left.splice(t,Number.POSITIVE_INFINITY);Qr(this.right,n.reverse())}else{const n=this.right.splice(this.left.length+this.right.length-t,Number.POSITIVE_INFINITY);Qr(this.left,n.reverse())}}}function Qr(e,t){let n=0;if(t.length<1e4)e.push(...t);else for(;n<t.length;)e.push(...t.slice(n,n+1e4)),n+=1e4}function kh(e){const t={};let n=-1,r,i,a,o,s,l,u;const d=new av(e);for(;++n<d.length;){for(;n in t;)n=t[n];if(r=d.get(n),n&&r[1].type==="chunkFlow"&&d.get(n-1)[1].type==="listItemPrefix"&&(l=r[1]._tokenizer.events,a=0,a<l.length&&l[a][1].type==="lineEndingBlank"&&(a+=2),a<l.length&&l[a][1].type==="content"))for(;++a<l.length&&l[a][1].type!=="content";)l[a][1].type==="chunkText"&&(l[a][1]._isInFirstContentOfListItem=!0,a++);if(r[0]==="enter")r[1].contentType&&(Object.assign(t,ov(d,n)),n=t[n],u=!0);else if(r[1]._container){for(a=n,i=void 0;a--;)if(o=d.get(a),o[1].type==="lineEnding"||o[1].type==="lineEndingBlank")o[0]==="enter"&&(i&&(d.get(i)[1].type="lineEndingBlank"),o[1].type="lineEnding",i=a);else if(!(o[1].type==="linePrefix"||o[1].type==="listItemIndent"))break;i&&(r[1].end={...d.get(i)[1].start},s=d.slice(i,n),s.unshift(r),d.splice(i,n-i+1,s))}}return ot(e,0,Number.POSITIVE_INFINITY,d.slice(0)),!u}function ov(e,t){const n=e.get(t)[1],r=e.get(t)[2];let i=t-1;const a=[];let o=n._tokenizer;o||(o=r.parser[n.contentType](n.start),n._contentTypeTextTrailing&&(o._contentTypeTextTrailing=!0));const s=o.events,l=[],u={};let d,c,p=-1,f=n,y=0,_=0;const S=[_];for(;f;){for(;e.get(++i)[1]!==f;);a.push(i),f._tokenizer||(d=r.sliceStream(f),f.next||d.push(null),c&&o.defineSkip(f.start),f._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=!0),o.write(d),f._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=void 0)),c=f,f=f.next}for(f=n;++p<s.length;)s[p][0]==="exit"&&s[p-1][0]==="enter"&&s[p][1].type===s[p-1][1].type&&s[p][1].start.line!==s[p][1].end.line&&(_=p+1,S.push(_),f._tokenizer=void 0,f.previous=void 0,f=f.next);for(o.events=[],f?(f._tokenizer=void 0,f.previous=void 0):S.pop(),p=S.length;p--;){const m=s.slice(S[p],S[p+1]),h=a.pop();l.push([h,h+m.length-1]),e.splice(h,2,m)}for(l.reverse(),p=-1;++p<l.length;)u[y+l[p][0]]=y+l[p][1],y+=l[p][1]-l[p][0]-1;return u}const sv={resolve:uv,tokenize:cv},lv={partial:!0,tokenize:dv};function uv(e){return kh(e),e}function cv(e,t){let n;return r;function r(s){return e.enter("content"),n=e.enter("chunkContent",{contentType:"content"}),i(s)}function i(s){return s===null?a(s):Q(s)?e.check(lv,o,a)(s):(e.consume(s),i)}function a(s){return e.exit("chunkContent"),e.exit("content"),t(s)}function o(s){return e.consume(s),e.exit("chunkContent"),n.next=e.enter("chunkContent",{contentType:"content",previous:n}),n=n.next,i}}function dv(e,t,n){const r=this;return i;function i(o){return e.exit("chunkContent"),e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),ne(e,a,"linePrefix")}function a(o){if(o===null||Q(o))return n(o);const s=r.events[r.events.length-1];return!r.parser.constructs.disable.null.includes("codeIndented")&&s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):e.interrupt(r.parser.constructs.flow,n,t)(o)}}function bh(e,t,n,r,i,a,o,s,l){const u=l||Number.POSITIVE_INFINITY;let d=0;return c;function c(m){return m===60?(e.enter(r),e.enter(i),e.enter(a),e.consume(m),e.exit(a),p):m===null||m===32||m===41||uo(m)?n(m):(e.enter(r),e.enter(o),e.enter(s),e.enter("chunkString",{contentType:"string"}),_(m))}function p(m){return m===62?(e.enter(a),e.consume(m),e.exit(a),e.exit(i),e.exit(r),t):(e.enter(s),e.enter("chunkString",{contentType:"string"}),f(m))}function f(m){return m===62?(e.exit("chunkString"),e.exit(s),p(m)):m===null||m===60||Q(m)?n(m):(e.consume(m),m===92?y:f)}function y(m){return m===60||m===62||m===92?(e.consume(m),f):f(m)}function _(m){return!d&&(m===null||m===41||ce(m))?(e.exit("chunkString"),e.exit(s),e.exit(o),e.exit(r),t(m)):d<u&&m===40?(e.consume(m),d++,_):m===41?(e.consume(m),d--,_):m===null||m===32||m===40||uo(m)?n(m):(e.consume(m),m===92?S:_)}function S(m){return m===40||m===41||m===92?(e.consume(m),_):_(m)}}function Sh(e,t,n,r,i,a){const o=this;let s=0,l;return u;function u(f){return e.enter(r),e.enter(i),e.consume(f),e.exit(i),e.enter(a),d}function d(f){return s>999||f===null||f===91||f===93&&!l||f===94&&!s&&"_hiddenFootnoteSupport"in o.parser.constructs?n(f):f===93?(e.exit(a),e.enter(i),e.consume(f),e.exit(i),e.exit(r),t):Q(f)?(e.enter("lineEnding"),e.consume(f),e.exit("lineEnding"),d):(e.enter("chunkString",{contentType:"string"}),c(f))}function c(f){return f===null||f===91||f===93||Q(f)||s++>999?(e.exit("chunkString"),d(f)):(e.consume(f),l||(l=!Z(f)),f===92?p:c)}function p(f){return f===91||f===92||f===93?(e.consume(f),s++,c):c(f)}}function Ch(e,t,n,r,i,a){let o;return s;function s(p){return p===34||p===39||p===40?(e.enter(r),e.enter(i),e.consume(p),e.exit(i),o=p===40?41:p,l):n(p)}function l(p){return p===o?(e.enter(i),e.consume(p),e.exit(i),e.exit(r),t):(e.enter(a),u(p))}function u(p){return p===o?(e.exit(a),l(o)):p===null?n(p):Q(p)?(e.enter("lineEnding"),e.consume(p),e.exit("lineEnding"),ne(e,u,"linePrefix")):(e.enter("chunkString",{contentType:"string"}),d(p))}function d(p){return p===o||p===null||Q(p)?(e.exit("chunkString"),u(p)):(e.consume(p),p===92?c:d)}function c(p){return p===o||p===92?(e.consume(p),d):d(p)}}function pi(e,t){let n;return r;function r(i){return Q(i)?(e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),n=!0,r):Z(i)?ne(e,r,n?"linePrefix":"lineSuffix")(i):t(i)}}const pv={name:"definition",tokenize:mv},fv={partial:!0,tokenize:hv};function mv(e,t,n){const r=this;let i;return a;function a(f){return e.enter("definition"),o(f)}function o(f){return Sh.call(r,e,s,n,"definitionLabel","definitionLabelMarker","definitionLabelString")(f)}function s(f){return i=Tt(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)),f===58?(e.enter("definitionMarker"),e.consume(f),e.exit("definitionMarker"),l):n(f)}function l(f){return ce(f)?pi(e,u)(f):u(f)}function u(f){return bh(e,d,n,"definitionDestination","definitionDestinationLiteral","definitionDestinationLiteralMarker","definitionDestinationRaw","definitionDestinationString")(f)}function d(f){return e.attempt(fv,c,c)(f)}function c(f){return Z(f)?ne(e,p,"whitespace")(f):p(f)}function p(f){return f===null||Q(f)?(e.exit("definition"),r.parser.defined.push(i),t(f)):n(f)}}function hv(e,t,n){return r;function r(s){return ce(s)?pi(e,i)(s):n(s)}function i(s){return Ch(e,a,n,"definitionTitle","definitionTitleMarker","definitionTitleString")(s)}function a(s){return Z(s)?ne(e,o,"whitespace")(s):o(s)}function o(s){return s===null||Q(s)?t(s):n(s)}}const gv={name:"hardBreakEscape",tokenize:yv};function yv(e,t,n){return r;function r(a){return e.enter("hardBreakEscape"),e.consume(a),i}function i(a){return Q(a)?(e.exit("hardBreakEscape"),t(a)):n(a)}}const xv={name:"headingAtx",resolve:_v,tokenize:vv};function _v(e,t){let n=e.length-2,r=3,i,a;return e[r][1].type==="whitespace"&&(r+=2),n-2>r&&e[n][1].type==="whitespace"&&(n-=2),e[n][1].type==="atxHeadingSequence"&&(r===n-1||n-4>r&&e[n-2][1].type==="whitespace")&&(n-=r+1===n?2:4),n>r&&(i={type:"atxHeadingText",start:e[r][1].start,end:e[n][1].end},a={type:"chunkText",start:e[r][1].start,end:e[n][1].end,contentType:"text"},ot(e,r,n-r+1,[["enter",i,t],["enter",a,t],["exit",a,t],["exit",i,t]])),e}function vv(e,t,n){let r=0;return i;function i(d){return e.enter("atxHeading"),a(d)}function a(d){return e.enter("atxHeadingSequence"),o(d)}function o(d){return d===35&&r++<6?(e.consume(d),o):d===null||ce(d)?(e.exit("atxHeadingSequence"),s(d)):n(d)}function s(d){return d===35?(e.enter("atxHeadingSequence"),l(d)):d===null||Q(d)?(e.exit("atxHeading"),t(d)):Z(d)?ne(e,s,"whitespace")(d):(e.enter("atxHeadingText"),u(d))}function l(d){return d===35?(e.consume(d),l):(e.exit("atxHeadingSequence"),s(d))}function u(d){return d===null||d===35||ce(d)?(e.exit("atxHeadingText"),s(d)):(e.consume(d),u)}}const wv=["address","article","aside","base","basefont","blockquote","body","caption","center","col","colgroup","dd","details","dialog","dir","div","dl","dt","fieldset","figcaption","figure","footer","form","frame","frameset","h1","h2","h3","h4","h5","h6","head","header","hr","html","iframe","legend","li","link","main","menu","menuitem","nav","noframes","ol","optgroup","option","p","param","search","section","summary","table","tbody","td","tfoot","th","thead","title","tr","track","ul"],Ud=["pre","script","style","textarea"],kv={concrete:!0,name:"htmlFlow",resolveTo:Cv,tokenize:zv},bv={partial:!0,tokenize:Pv},Sv={partial:!0,tokenize:Ev};function Cv(e){let t=e.length;for(;t--&&!(e[t][0]==="enter"&&e[t][1].type==="htmlFlow"););return t>1&&e[t-2][1].type==="linePrefix"&&(e[t][1].start=e[t-2][1].start,e[t+1][1].start=e[t-2][1].start,e.splice(t-2,2)),e}function zv(e,t,n){const r=this;let i,a,o,s,l;return u;function u(w){return d(w)}function d(w){return e.enter("htmlFlow"),e.enter("htmlFlowData"),e.consume(w),c}function c(w){return w===33?(e.consume(w),p):w===47?(e.consume(w),a=!0,_):w===63?(e.consume(w),i=3,r.interrupt?t:v):Ue(w)?(e.consume(w),o=String.fromCharCode(w),S):n(w)}function p(w){return w===45?(e.consume(w),i=2,f):w===91?(e.consume(w),i=5,s=0,y):Ue(w)?(e.consume(w),i=4,r.interrupt?t:v):n(w)}function f(w){return w===45?(e.consume(w),r.interrupt?t:v):n(w)}function y(w){const ie="CDATA[";return w===ie.charCodeAt(s++)?(e.consume(w),s===ie.length?r.interrupt?t:W:y):n(w)}function _(w){return Ue(w)?(e.consume(w),o=String.fromCharCode(w),S):n(w)}function S(w){if(w===null||w===47||w===62||ce(w)){const ie=w===47,ke=o.toLowerCase();return!ie&&!a&&Ud.includes(ke)?(i=1,r.interrupt?t(w):W(w)):wv.includes(o.toLowerCase())?(i=6,ie?(e.consume(w),m):r.interrupt?t(w):W(w)):(i=7,r.interrupt&&!r.parser.lazy[r.now().line]?n(w):a?h(w):g(w))}return w===45||Me(w)?(e.consume(w),o+=String.fromCharCode(w),S):n(w)}function m(w){return w===62?(e.consume(w),r.interrupt?t:W):n(w)}function h(w){return Z(w)?(e.consume(w),h):k(w)}function g(w){return w===47?(e.consume(w),k):w===58||w===95||Ue(w)?(e.consume(w),C):Z(w)?(e.consume(w),g):k(w)}function C(w){return w===45||w===46||w===58||w===95||Me(w)?(e.consume(w),C):E(w)}function E(w){return w===61?(e.consume(w),b):Z(w)?(e.consume(w),E):g(w)}function b(w){return w===null||w===60||w===61||w===62||w===96?n(w):w===34||w===39?(e.consume(w),l=w,P):Z(w)?(e.consume(w),b):T(w)}function P(w){return w===l?(e.consume(w),l=null,D):w===null||Q(w)?n(w):(e.consume(w),P)}function T(w){return w===null||w===34||w===39||w===47||w===60||w===61||w===62||w===96||ce(w)?E(w):(e.consume(w),T)}function D(w){return w===47||w===62||Z(w)?g(w):n(w)}function k(w){return w===62?(e.consume(w),I):n(w)}function I(w){return w===null||Q(w)?W(w):Z(w)?(e.consume(w),I):n(w)}function W(w){return w===45&&i===2?(e.consume(w),X):w===60&&i===1?(e.consume(w),re):w===62&&i===4?(e.consume(w),B):w===63&&i===3?(e.consume(w),v):w===93&&i===5?(e.consume(w),V):Q(w)&&(i===6||i===7)?(e.exit("htmlFlowData"),e.check(bv,$,K)(w)):w===null||Q(w)?(e.exit("htmlFlowData"),K(w)):(e.consume(w),W)}function K(w){return e.check(Sv,te,$)(w)}function te(w){return e.enter("lineEnding"),e.consume(w),e.exit("lineEnding"),F}function F(w){return w===null||Q(w)?K(w):(e.enter("htmlFlowData"),W(w))}function X(w){return w===45?(e.consume(w),v):W(w)}function re(w){return w===47?(e.consume(w),o="",j):W(w)}function j(w){if(w===62){const ie=o.toLowerCase();return Ud.includes(ie)?(e.consume(w),B):W(w)}return Ue(w)&&o.length<8?(e.consume(w),o+=String.fromCharCode(w),j):W(w)}function V(w){return w===93?(e.consume(w),v):W(w)}function v(w){return w===62?(e.consume(w),B):w===45&&i===2?(e.consume(w),v):W(w)}function B(w){return w===null||Q(w)?(e.exit("htmlFlowData"),$(w)):(e.consume(w),B)}function $(w){return e.exit("htmlFlow"),t(w)}}function Ev(e,t,n){const r=this;return i;function i(o){return Q(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a):n(o)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}function Pv(e,t,n){return r;function r(i){return e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),e.attempt(Xi,t,n)}}const Nv={name:"htmlText",tokenize:Tv};function Tv(e,t,n){const r=this;let i,a,o;return s;function s(v){return e.enter("htmlText"),e.enter("htmlTextData"),e.consume(v),l}function l(v){return v===33?(e.consume(v),u):v===47?(e.consume(v),E):v===63?(e.consume(v),g):Ue(v)?(e.consume(v),T):n(v)}function u(v){return v===45?(e.consume(v),d):v===91?(e.consume(v),a=0,y):Ue(v)?(e.consume(v),h):n(v)}function d(v){return v===45?(e.consume(v),f):n(v)}function c(v){return v===null?n(v):v===45?(e.consume(v),p):Q(v)?(o=c,re(v)):(e.consume(v),c)}function p(v){return v===45?(e.consume(v),f):c(v)}function f(v){return v===62?X(v):v===45?p(v):c(v)}function y(v){const B="CDATA[";return v===B.charCodeAt(a++)?(e.consume(v),a===B.length?_:y):n(v)}function _(v){return v===null?n(v):v===93?(e.consume(v),S):Q(v)?(o=_,re(v)):(e.consume(v),_)}function S(v){return v===93?(e.consume(v),m):_(v)}function m(v){return v===62?X(v):v===93?(e.consume(v),m):_(v)}function h(v){return v===null||v===62?X(v):Q(v)?(o=h,re(v)):(e.consume(v),h)}function g(v){return v===null?n(v):v===63?(e.consume(v),C):Q(v)?(o=g,re(v)):(e.consume(v),g)}function C(v){return v===62?X(v):g(v)}function E(v){return Ue(v)?(e.consume(v),b):n(v)}function b(v){return v===45||Me(v)?(e.consume(v),b):P(v)}function P(v){return Q(v)?(o=P,re(v)):Z(v)?(e.consume(v),P):X(v)}function T(v){return v===45||Me(v)?(e.consume(v),T):v===47||v===62||ce(v)?D(v):n(v)}function D(v){return v===47?(e.consume(v),X):v===58||v===95||Ue(v)?(e.consume(v),k):Q(v)?(o=D,re(v)):Z(v)?(e.consume(v),D):X(v)}function k(v){return v===45||v===46||v===58||v===95||Me(v)?(e.consume(v),k):I(v)}function I(v){return v===61?(e.consume(v),W):Q(v)?(o=I,re(v)):Z(v)?(e.consume(v),I):D(v)}function W(v){return v===null||v===60||v===61||v===62||v===96?n(v):v===34||v===39?(e.consume(v),i=v,K):Q(v)?(o=W,re(v)):Z(v)?(e.consume(v),W):(e.consume(v),te)}function K(v){return v===i?(e.consume(v),i=void 0,F):v===null?n(v):Q(v)?(o=K,re(v)):(e.consume(v),K)}function te(v){return v===null||v===34||v===39||v===60||v===61||v===96?n(v):v===47||v===62||ce(v)?D(v):(e.consume(v),te)}function F(v){return v===47||v===62||ce(v)?D(v):n(v)}function X(v){return v===62?(e.consume(v),e.exit("htmlTextData"),e.exit("htmlText"),t):n(v)}function re(v){return e.exit("htmlTextData"),e.enter("lineEnding"),e.consume(v),e.exit("lineEnding"),j}function j(v){return Z(v)?ne(e,V,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(v):V(v)}function V(v){return e.enter("htmlTextData"),o(v)}}const Ku={name:"labelEnd",resolveAll:jv,resolveTo:Iv,tokenize:Dv},Lv={tokenize:Fv},Av={tokenize:Mv},Rv={tokenize:Ov};function jv(e){let t=-1;const n=[];for(;++t<e.length;){const r=e[t][1];if(n.push(e[t]),r.type==="labelImage"||r.type==="labelLink"||r.type==="labelEnd"){const i=r.type==="labelImage"?4:2;r.type="data",t+=i}}return e.length!==n.length&&ot(e,0,e.length,n),e}function Iv(e,t){let n=e.length,r=0,i,a,o,s;for(;n--;)if(i=e[n][1],a){if(i.type==="link"||i.type==="labelLink"&&i._inactive)break;e[n][0]==="enter"&&i.type==="labelLink"&&(i._inactive=!0)}else if(o){if(e[n][0]==="enter"&&(i.type==="labelImage"||i.type==="labelLink")&&!i._balanced&&(a=n,i.type!=="labelLink")){r=2;break}}else i.type==="labelEnd"&&(o=n);const l={type:e[a][1].type==="labelLink"?"link":"image",start:{...e[a][1].start},end:{...e[e.length-1][1].end}},u={type:"label",start:{...e[a][1].start},end:{...e[o][1].end}},d={type:"labelText",start:{...e[a+r+2][1].end},end:{...e[o-2][1].start}};return s=[["enter",l,t],["enter",u,t]],s=mt(s,e.slice(a+1,a+r+3)),s=mt(s,[["enter",d,t]]),s=mt(s,Fo(t.parser.constructs.insideSpan.null,e.slice(a+r+4,o-3),t)),s=mt(s,[["exit",d,t],e[o-2],e[o-1],["exit",u,t]]),s=mt(s,e.slice(o+1)),s=mt(s,[["exit",l,t]]),ot(e,a,e.length,s),e}function Dv(e,t,n){const r=this;let i=r.events.length,a,o;for(;i--;)if((r.events[i][1].type==="labelImage"||r.events[i][1].type==="labelLink")&&!r.events[i][1]._balanced){a=r.events[i][1];break}return s;function s(p){return a?a._inactive?c(p):(o=r.parser.defined.includes(Tt(r.sliceSerialize({start:a.end,end:r.now()}))),e.enter("labelEnd"),e.enter("labelMarker"),e.consume(p),e.exit("labelMarker"),e.exit("labelEnd"),l):n(p)}function l(p){return p===40?e.attempt(Lv,d,o?d:c)(p):p===91?e.attempt(Av,d,o?u:c)(p):o?d(p):c(p)}function u(p){return e.attempt(Rv,d,c)(p)}function d(p){return t(p)}function c(p){return a._balanced=!0,n(p)}}function Fv(e,t,n){return r;function r(c){return e.enter("resource"),e.enter("resourceMarker"),e.consume(c),e.exit("resourceMarker"),i}function i(c){return ce(c)?pi(e,a)(c):a(c)}function a(c){return c===41?d(c):bh(e,o,s,"resourceDestination","resourceDestinationLiteral","resourceDestinationLiteralMarker","resourceDestinationRaw","resourceDestinationString",32)(c)}function o(c){return ce(c)?pi(e,l)(c):d(c)}function s(c){return n(c)}function l(c){return c===34||c===39||c===40?Ch(e,u,n,"resourceTitle","resourceTitleMarker","resourceTitleString")(c):d(c)}function u(c){return ce(c)?pi(e,d)(c):d(c)}function d(c){return c===41?(e.enter("resourceMarker"),e.consume(c),e.exit("resourceMarker"),e.exit("resource"),t):n(c)}}function Mv(e,t,n){const r=this;return i;function i(s){return Sh.call(r,e,a,o,"reference","referenceMarker","referenceString")(s)}function a(s){return r.parser.defined.includes(Tt(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)))?t(s):n(s)}function o(s){return n(s)}}function Ov(e,t,n){return r;function r(a){return e.enter("reference"),e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),i}function i(a){return a===93?(e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),e.exit("reference"),t):n(a)}}const qv={name:"labelStartImage",resolveAll:Ku.resolveAll,tokenize:Wv};function Wv(e,t,n){const r=this;return i;function i(s){return e.enter("labelImage"),e.enter("labelImageMarker"),e.consume(s),e.exit("labelImageMarker"),a}function a(s){return s===91?(e.enter("labelMarker"),e.consume(s),e.exit("labelMarker"),e.exit("labelImage"),o):n(s)}function o(s){return s===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(s):t(s)}}const Bv={name:"labelStartLink",resolveAll:Ku.resolveAll,tokenize:Uv};function Uv(e,t,n){const r=this;return i;function i(o){return e.enter("labelLink"),e.enter("labelMarker"),e.consume(o),e.exit("labelMarker"),e.exit("labelLink"),a}function a(o){return o===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(o):t(o)}}const _s={name:"lineEnding",tokenize:Vv};function Vv(e,t){return n;function n(r){return e.enter("lineEnding"),e.consume(r),e.exit("lineEnding"),ne(e,t,"linePrefix")}}const ja={name:"thematicBreak",tokenize:Hv};function Hv(e,t,n){let r=0,i;return a;function a(u){return e.enter("thematicBreak"),o(u)}function o(u){return i=u,s(u)}function s(u){return u===i?(e.enter("thematicBreakSequence"),l(u)):r>=3&&(u===null||Q(u))?(e.exit("thematicBreak"),t(u)):n(u)}function l(u){return u===i?(e.consume(u),r++,l):(e.exit("thematicBreakSequence"),Z(u)?ne(e,s,"whitespace")(u):s(u))}}const Qe={continuation:{tokenize:Kv},exit:Gv,name:"list",tokenize:Xv},$v={partial:!0,tokenize:Zv},Qv={partial:!0,tokenize:Yv};function Xv(e,t,n){const r=this,i=r.events[r.events.length-1];let a=i&&i[1].type==="linePrefix"?i[2].sliceSerialize(i[1],!0).length:0,o=0;return s;function s(f){const y=r.containerState.type||(f===42||f===43||f===45?"listUnordered":"listOrdered");if(y==="listUnordered"?!r.containerState.marker||f===r.containerState.marker:Ml(f)){if(r.containerState.type||(r.containerState.type=y,e.enter(y,{_container:!0})),y==="listUnordered")return e.enter("listItemPrefix"),f===42||f===45?e.check(ja,n,u)(f):u(f);if(!r.interrupt||f===49)return e.enter("listItemPrefix"),e.enter("listItemValue"),l(f)}return n(f)}function l(f){return Ml(f)&&++o<10?(e.consume(f),l):(!r.interrupt||o<2)&&(r.containerState.marker?f===r.containerState.marker:f===41||f===46)?(e.exit("listItemValue"),u(f)):n(f)}function u(f){return e.enter("listItemMarker"),e.consume(f),e.exit("listItemMarker"),r.containerState.marker=r.containerState.marker||f,e.check(Xi,r.interrupt?n:d,e.attempt($v,p,c))}function d(f){return r.containerState.initialBlankLine=!0,a++,p(f)}function c(f){return Z(f)?(e.enter("listItemPrefixWhitespace"),e.consume(f),e.exit("listItemPrefixWhitespace"),p):n(f)}function p(f){return r.containerState.size=a+r.sliceSerialize(e.exit("listItemPrefix"),!0).length,t(f)}}function Kv(e,t,n){const r=this;return r.containerState._closeFlow=void 0,e.check(Xi,i,a);function i(s){return r.containerState.furtherBlankLines=r.containerState.furtherBlankLines||r.containerState.initialBlankLine,ne(e,t,"listItemIndent",r.containerState.size+1)(s)}function a(s){return r.containerState.furtherBlankLines||!Z(s)?(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,o(s)):(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,e.attempt(Qv,t,o)(s))}function o(s){return r.containerState._closeFlow=!0,r.interrupt=void 0,ne(e,e.attempt(Qe,t,n),"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(s)}}function Yv(e,t,n){const r=this;return ne(e,i,"listItemIndent",r.containerState.size+1);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="listItemIndent"&&o[2].sliceSerialize(o[1],!0).length===r.containerState.size?t(a):n(a)}}function Gv(e){e.exit(this.containerState.type)}function Zv(e,t,n){const r=this;return ne(e,i,"listItemPrefixWhitespace",r.parser.constructs.disable.null.includes("codeIndented")?void 0:5);function i(a){const o=r.events[r.events.length-1];return!Z(a)&&o&&o[1].type==="listItemPrefixWhitespace"?t(a):n(a)}}const Vd={name:"setextUnderline",resolveTo:Jv,tokenize:ew};function Jv(e,t){let n=e.length,r,i,a;for(;n--;)if(e[n][0]==="enter"){if(e[n][1].type==="content"){r=n;break}e[n][1].type==="paragraph"&&(i=n)}else e[n][1].type==="content"&&e.splice(n,1),!a&&e[n][1].type==="definition"&&(a=n);const o={type:"setextHeading",start:{...e[r][1].start},end:{...e[e.length-1][1].end}};return e[i][1].type="setextHeadingText",a?(e.splice(i,0,["enter",o,t]),e.splice(a+1,0,["exit",e[r][1],t]),e[r][1].end={...e[a][1].end}):e[r][1]=o,e.push(["exit",o,t]),e}function ew(e,t,n){const r=this;let i;return a;function a(u){let d=r.events.length,c;for(;d--;)if(r.events[d][1].type!=="lineEnding"&&r.events[d][1].type!=="linePrefix"&&r.events[d][1].type!=="content"){c=r.events[d][1].type==="paragraph";break}return!r.parser.lazy[r.now().line]&&(r.interrupt||c)?(e.enter("setextHeadingLine"),i=u,o(u)):n(u)}function o(u){return e.enter("setextHeadingLineSequence"),s(u)}function s(u){return u===i?(e.consume(u),s):(e.exit("setextHeadingLineSequence"),Z(u)?ne(e,l,"lineSuffix")(u):l(u))}function l(u){return u===null||Q(u)?(e.exit("setextHeadingLine"),t(u)):n(u)}}const tw={tokenize:nw};function nw(e){const t=this,n=e.attempt(Xi,r,e.attempt(this.parser.constructs.flowInitial,i,ne(e,e.attempt(this.parser.constructs.flow,i,e.attempt(sv,i)),"linePrefix")));return n;function r(a){if(a===null){e.consume(a);return}return e.enter("lineEndingBlank"),e.consume(a),e.exit("lineEndingBlank"),t.currentConstruct=void 0,n}function i(a){if(a===null){e.consume(a);return}return e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),t.currentConstruct=void 0,n}}const rw={resolveAll:Eh()},iw=zh("string"),aw=zh("text");function zh(e){return{resolveAll:Eh(e==="text"?ow:void 0),tokenize:t};function t(n){const r=this,i=this.parser.constructs[e],a=n.attempt(i,o,s);return o;function o(d){return u(d)?a(d):s(d)}function s(d){if(d===null){n.consume(d);return}return n.enter("data"),n.consume(d),l}function l(d){return u(d)?(n.exit("data"),a(d)):(n.consume(d),l)}function u(d){if(d===null)return!0;const c=i[d];let p=-1;if(c)for(;++p<c.length;){const f=c[p];if(!f.previous||f.previous.call(r,r.previous))return!0}return!1}}}function Eh(e){return t;function t(n,r){let i=-1,a;for(;++i<=n.length;)a===void 0?n[i]&&n[i][1].type==="data"&&(a=i,i++):(!n[i]||n[i][1].type!=="data")&&(i!==a+2&&(n[a][1].end=n[i-1][1].end,n.splice(a+2,i-a-2),i=a+2),a=void 0);return e?e(n,r):n}}function ow(e,t){let n=0;for(;++n<=e.length;)if((n===e.length||e[n][1].type==="lineEnding")&&e[n-1][1].type==="data"){const r=e[n-1][1],i=t.sliceStream(r);let a=i.length,o=-1,s=0,l;for(;a--;){const u=i[a];if(typeof u=="string"){for(o=u.length;u.charCodeAt(o-1)===32;)s++,o--;if(o)break;o=-1}else if(u===-2)l=!0,s++;else if(u!==-1){a++;break}}if(t._contentTypeTextTrailing&&n===e.length&&(s=0),s){const u={type:n===e.length||l||s<2?"lineSuffix":"hardBreakTrailing",start:{_bufferIndex:a?o:r.start._bufferIndex+o,_index:r.start._index+a,line:r.end.line,column:r.end.column-s,offset:r.end.offset-s},end:{...r.end}};r.end={...u.start},r.start.offset===r.end.offset?Object.assign(r,u):(e.splice(n,0,["enter",u,t],["exit",u,t]),n+=2)}n++}return e}const sw={42:Qe,43:Qe,45:Qe,48:Qe,49:Qe,50:Qe,51:Qe,52:Qe,53:Qe,54:Qe,55:Qe,56:Qe,57:Qe,62:_h},lw={91:pv},uw={[-2]:xs,[-1]:xs,32:xs},cw={35:xv,42:ja,45:[Vd,ja],60:kv,61:Vd,95:ja,96:Bd,126:Bd},dw={38:wh,92:vh},pw={[-5]:_s,[-4]:_s,[-3]:_s,33:qv,38:wh,42:Ol,60:[B_,Nv],91:Bv,92:[gv,vh],93:Ku,95:Ol,96:tv},fw={null:[Ol,rw]},mw={null:[42,95]},hw={null:[]},gw=Object.freeze(Object.defineProperty({__proto__:null,attentionMarkers:mw,contentInitial:lw,disable:hw,document:sw,flow:cw,flowInitial:uw,insideSpan:fw,string:dw,text:pw},Symbol.toStringTag,{value:"Module"}));function yw(e,t,n){let r={_bufferIndex:-1,_index:0,line:n&&n.line||1,column:n&&n.column||1,offset:n&&n.offset||0};const i={},a=[];let o=[],s=[];const l={attempt:P(E),check:P(b),consume:h,enter:g,exit:C,interrupt:P(b,{interrupt:!0})},u={code:null,containerState:{},defineSkip:_,events:[],now:y,parser:e,previous:null,sliceSerialize:p,sliceStream:f,write:c};let d=t.tokenize.call(u,l);return t.resolveAll&&a.push(t),u;function c(I){return o=mt(o,I),S(),o[o.length-1]!==null?[]:(T(t,0),u.events=Fo(a,u.events,u),u.events)}function p(I,W){return _w(f(I),W)}function f(I){return xw(o,I)}function y(){const{_bufferIndex:I,_index:W,line:K,column:te,offset:F}=r;return{_bufferIndex:I,_index:W,line:K,column:te,offset:F}}function _(I){i[I.line]=I.column,k()}function S(){let I;for(;r._index<o.length;){const W=o[r._index];if(typeof W=="string")for(I=r._index,r._bufferIndex<0&&(r._bufferIndex=0);r._index===I&&r._bufferIndex<W.length;)m(W.charCodeAt(r._bufferIndex));else m(W)}}function m(I){d=d(I)}function h(I){Q(I)?(r.line++,r.column=1,r.offset+=I===-3?2:1,k()):I!==-1&&(r.column++,r.offset++),r._bufferIndex<0?r._index++:(r._bufferIndex++,r._bufferIndex===o[r._index].length&&(r._bufferIndex=-1,r._index++)),u.previous=I}function g(I,W){const K=W||{};return K.type=I,K.start=y(),u.events.push(["enter",K,u]),s.push(K),K}function C(I){const W=s.pop();return W.end=y(),u.events.push(["exit",W,u]),W}function E(I,W){T(I,W.from)}function b(I,W){W.restore()}function P(I,W){return K;function K(te,F,X){let re,j,V,v;return Array.isArray(te)?$(te):"tokenize"in te?$([te]):B(te);function B(le){return ct;function ct(q){const L=q!==null&&le[q],M=q!==null&&le.null,H=[...Array.isArray(L)?L:L?[L]:[],...Array.isArray(M)?M:M?[M]:[]];return $(H)(q)}}function $(le){return re=le,j=0,le.length===0?X:w(le[j])}function w(le){return ct;function ct(q){return v=D(),V=le,le.partial||(u.currentConstruct=le),le.name&&u.parser.constructs.disable.null.includes(le.name)?ke():le.tokenize.call(W?Object.assign(Object.create(u),W):u,l,ie,ke)(q)}}function ie(le){return I(V,v),F}function ke(le){return v.restore(),++j<re.length?w(re[j]):X}}}function T(I,W){I.resolveAll&&!a.includes(I)&&a.push(I),I.resolve&&ot(u.events,W,u.events.length-W,I.resolve(u.events.slice(W),u)),I.resolveTo&&(u.events=I.resolveTo(u.events,u))}function D(){const I=y(),W=u.previous,K=u.currentConstruct,te=u.events.length,F=Array.from(s);return{from:te,restore:X};function X(){r=I,u.previous=W,u.currentConstruct=K,u.events.length=te,s=F,k()}}function k(){r.line in i&&r.column<2&&(r.column=i[r.line],r.offset+=i[r.line]-1)}}function xw(e,t){const n=t.start._index,r=t.start._bufferIndex,i=t.end._index,a=t.end._bufferIndex;let o;if(n===i)o=[e[n].slice(r,a)];else{if(o=e.slice(n,i),r>-1){const s=o[0];typeof s=="string"?o[0]=s.slice(r):o.shift()}a>0&&o.push(e[i].slice(0,a))}return o}function _w(e,t){let n=-1;const r=[];let i;for(;++n<e.length;){const a=e[n];let o;if(typeof a=="string")o=a;else switch(a){case-5:{o="\r";break}case-4:{o=`
`;break}case-3:{o=`\r
`;break}case-2:{o=t?" ":"	";break}case-1:{if(!t&&i)continue;o=" ";break}default:o=String.fromCharCode(a)}i=a===-2,r.push(o)}return r.join("")}function vw(e){const r={constructs:yh([gw,...(e||{}).extensions||[]]),content:i(I_),defined:[],document:i(F_),flow:i(tw),lazy:{},string:i(iw),text:i(aw)};return r;function i(a){return o;function o(s){return yw(r,a,s)}}}function ww(e){for(;!kh(e););return e}const Hd=/[\0\t\n\r]/g;function kw(){let e=1,t="",n=!0,r;return i;function i(a,o,s){const l=[];let u,d,c,p,f;for(a=t+(typeof a=="string"?a.toString():new TextDecoder(o||void 0).decode(a)),c=0,t="",n&&(a.charCodeAt(0)===65279&&c++,n=void 0);c<a.length;){if(Hd.lastIndex=c,u=Hd.exec(a),p=u&&u.index!==void 0?u.index:a.length,f=a.charCodeAt(p),!u){t=a.slice(c);break}if(f===10&&c===p&&r)l.push(-3),r=void 0;else switch(r&&(l.push(-5),r=void 0),c<p&&(l.push(a.slice(c,p)),e+=p-c),f){case 0:{l.push(65533),e++;break}case 9:{for(d=Math.ceil(e/4)*4,l.push(-2);e++<d;)l.push(-1);break}case 10:{l.push(-4),e=1;break}default:r=!0,e=1}c=p+1}return s&&(r&&l.push(-5),t&&l.push(t),l.push(null)),l}}const bw=/\\([!-/:-@[-`{-~])|&(#(?:\d{1,7}|x[\da-f]{1,6})|[\da-z]{1,31});/gi;function Sw(e){return e.replace(bw,Cw)}function Cw(e,t,n){if(t)return t;if(n.charCodeAt(0)===35){const i=n.charCodeAt(1),a=i===120||i===88;return xh(n.slice(a?2:1),a?16:10)}return Xu(n)||e}const Ph={}.hasOwnProperty;function zw(e,t,n){return typeof t!="string"&&(n=t,t=void 0),Ew(n)(ww(vw(n).document().write(kw()(e,t,!0))))}function Ew(e){const t={transforms:[],canContainEols:["emphasis","fragment","heading","paragraph","strong"],enter:{autolink:a(oc),autolinkProtocol:D,autolinkEmail:D,atxHeading:a(At),blockQuote:a(M),characterEscape:D,characterReference:D,codeFenced:a(H),codeFencedFenceInfo:o,codeFencedFenceMeta:o,codeIndented:a(H,o),codeText:a(ze,o),codeTextData:D,data:D,codeFlowValue:D,definition:a(_e),definitionDestinationString:o,definitionLabelString:o,definitionTitleString:o,emphasis:a(We),hardBreakEscape:a(Tn),hardBreakTrailing:a(Tn),htmlFlow:a(Ln,o),htmlFlowData:D,htmlText:a(Ln,o),htmlTextData:D,image:a(Cg),label:o,link:a(oc),listItem:a(zg),listItemValue:p,listOrdered:a(sc,c),listUnordered:a(sc),paragraph:a(Eg),reference:w,referenceString:o,resourceDestinationString:o,resourceTitleString:o,setextHeading:a(At),strong:a(Pg),thematicBreak:a(Tg)},exit:{atxHeading:l(),atxHeadingSequence:E,autolink:l(),autolinkEmail:L,autolinkProtocol:q,blockQuote:l(),characterEscapeValue:k,characterReferenceMarkerHexadecimal:ke,characterReferenceMarkerNumeric:ke,characterReferenceValue:le,characterReference:ct,codeFenced:l(S),codeFencedFence:_,codeFencedFenceInfo:f,codeFencedFenceMeta:y,codeFlowValue:k,codeIndented:l(m),codeText:l(F),codeTextData:k,data:k,definition:l(),definitionDestinationString:C,definitionLabelString:h,definitionTitleString:g,emphasis:l(),hardBreakEscape:l(W),hardBreakTrailing:l(W),htmlFlow:l(K),htmlFlowData:k,htmlText:l(te),htmlTextData:k,image:l(re),label:V,labelText:j,lineEnding:I,link:l(X),listItem:l(),listOrdered:l(),listUnordered:l(),paragraph:l(),referenceString:ie,resourceDestinationString:v,resourceTitleString:B,resource:$,setextHeading:l(T),setextHeadingLineSequence:P,setextHeadingText:b,strong:l(),thematicBreak:l()}};Nh(t,(e||{}).mdastExtensions||[]);const n={};return r;function r(N){let O={type:"root",children:[]};const Y={stack:[O],tokenStack:[],config:t,enter:s,exit:u,buffer:o,resume:d,data:n},ee=[];let ue=-1;for(;++ue<N.length;)if(N[ue][1].type==="listOrdered"||N[ue][1].type==="listUnordered")if(N[ue][0]==="enter")ee.push(ue);else{const _t=ee.pop();ue=i(N,_t,ue)}for(ue=-1;++ue<N.length;){const _t=t[N[ue][0]];Ph.call(_t,N[ue][1].type)&&_t[N[ue][1].type].call(Object.assign({sliceSerialize:N[ue][2].sliceSerialize},Y),N[ue][1])}if(Y.tokenStack.length>0){const _t=Y.tokenStack[Y.tokenStack.length-1];(_t[1]||$d).call(Y,void 0,_t[0])}for(O.position={start:rn(N.length>0?N[0][1].start:{line:1,column:1,offset:0}),end:rn(N.length>0?N[N.length-2][1].end:{line:1,column:1,offset:0})},ue=-1;++ue<t.transforms.length;)O=t.transforms[ue](O)||O;return O}function i(N,O,Y){let ee=O-1,ue=-1,_t=!1,An,Bt,Fr,Mr;for(;++ee<=Y;){const et=N[ee];switch(et[1].type){case"listUnordered":case"listOrdered":case"blockQuote":{et[0]==="enter"?ue++:ue--,Mr=void 0;break}case"lineEndingBlank":{et[0]==="enter"&&(An&&!Mr&&!ue&&!Fr&&(Fr=ee),Mr=void 0);break}case"linePrefix":case"listItemValue":case"listItemMarker":case"listItemPrefix":case"listItemPrefixWhitespace":break;default:Mr=void 0}if(!ue&&et[0]==="enter"&&et[1].type==="listItemPrefix"||ue===-1&&et[0]==="exit"&&(et[1].type==="listUnordered"||et[1].type==="listOrdered")){if(An){let Gn=ee;for(Bt=void 0;Gn--;){const Ut=N[Gn];if(Ut[1].type==="lineEnding"||Ut[1].type==="lineEndingBlank"){if(Ut[0]==="exit")continue;Bt&&(N[Bt][1].type="lineEndingBlank",_t=!0),Ut[1].type="lineEnding",Bt=Gn}else if(!(Ut[1].type==="linePrefix"||Ut[1].type==="blockQuotePrefix"||Ut[1].type==="blockQuotePrefixWhitespace"||Ut[1].type==="blockQuoteMarker"||Ut[1].type==="listItemIndent"))break}Fr&&(!Bt||Fr<Bt)&&(An._spread=!0),An.end=Object.assign({},Bt?N[Bt][1].start:et[1].end),N.splice(Bt||ee,0,["exit",An,et[2]]),ee++,Y++}if(et[1].type==="listItemPrefix"){const Gn={type:"listItem",_spread:!1,start:Object.assign({},et[1].start),end:void 0};An=Gn,N.splice(ee,0,["enter",Gn,et[2]]),ee++,Y++,Fr=void 0,Mr=!0}}}return N[O][1]._spread=_t,Y}function a(N,O){return Y;function Y(ee){s.call(this,N(ee),ee),O&&O.call(this,ee)}}function o(){this.stack.push({type:"fragment",children:[]})}function s(N,O,Y){this.stack[this.stack.length-1].children.push(N),this.stack.push(N),this.tokenStack.push([O,Y||void 0]),N.position={start:rn(O.start),end:void 0}}function l(N){return O;function O(Y){N&&N.call(this,Y),u.call(this,Y)}}function u(N,O){const Y=this.stack.pop(),ee=this.tokenStack.pop();if(ee)ee[0].type!==N.type&&(O?O.call(this,N,ee[0]):(ee[1]||$d).call(this,N,ee[0]));else throw new Error("Cannot close `"+N.type+"` ("+di({start:N.start,end:N.end})+"): it’s not open");Y.position.end=rn(N.end)}function d(){return Qu(this.stack.pop())}function c(){this.data.expectingFirstListItemValue=!0}function p(N){if(this.data.expectingFirstListItemValue){const O=this.stack[this.stack.length-2];O.start=Number.parseInt(this.sliceSerialize(N),10),this.data.expectingFirstListItemValue=void 0}}function f(){const N=this.resume(),O=this.stack[this.stack.length-1];O.lang=N}function y(){const N=this.resume(),O=this.stack[this.stack.length-1];O.meta=N}function _(){this.data.flowCodeInside||(this.buffer(),this.data.flowCodeInside=!0)}function S(){const N=this.resume(),O=this.stack[this.stack.length-1];O.value=N.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g,""),this.data.flowCodeInside=void 0}function m(){const N=this.resume(),O=this.stack[this.stack.length-1];O.value=N.replace(/(\r?\n|\r)$/g,"")}function h(N){const O=this.resume(),Y=this.stack[this.stack.length-1];Y.label=O,Y.identifier=Tt(this.sliceSerialize(N)).toLowerCase()}function g(){const N=this.resume(),O=this.stack[this.stack.length-1];O.title=N}function C(){const N=this.resume(),O=this.stack[this.stack.length-1];O.url=N}function E(N){const O=this.stack[this.stack.length-1];if(!O.depth){const Y=this.sliceSerialize(N).length;O.depth=Y}}function b(){this.data.setextHeadingSlurpLineEnding=!0}function P(N){const O=this.stack[this.stack.length-1];O.depth=this.sliceSerialize(N).codePointAt(0)===61?1:2}function T(){this.data.setextHeadingSlurpLineEnding=void 0}function D(N){const Y=this.stack[this.stack.length-1].children;let ee=Y[Y.length-1];(!ee||ee.type!=="text")&&(ee=Ng(),ee.position={start:rn(N.start),end:void 0},Y.push(ee)),this.stack.push(ee)}function k(N){const O=this.stack.pop();O.value+=this.sliceSerialize(N),O.position.end=rn(N.end)}function I(N){const O=this.stack[this.stack.length-1];if(this.data.atHardBreak){const Y=O.children[O.children.length-1];Y.position.end=rn(N.end),this.data.atHardBreak=void 0;return}!this.data.setextHeadingSlurpLineEnding&&t.canContainEols.includes(O.type)&&(D.call(this,N),k.call(this,N))}function W(){this.data.atHardBreak=!0}function K(){const N=this.resume(),O=this.stack[this.stack.length-1];O.value=N}function te(){const N=this.resume(),O=this.stack[this.stack.length-1];O.value=N}function F(){const N=this.resume(),O=this.stack[this.stack.length-1];O.value=N}function X(){const N=this.stack[this.stack.length-1];if(this.data.inReference){const O=this.data.referenceType||"shortcut";N.type+="Reference",N.referenceType=O,delete N.url,delete N.title}else delete N.identifier,delete N.label;this.data.referenceType=void 0}function re(){const N=this.stack[this.stack.length-1];if(this.data.inReference){const O=this.data.referenceType||"shortcut";N.type+="Reference",N.referenceType=O,delete N.url,delete N.title}else delete N.identifier,delete N.label;this.data.referenceType=void 0}function j(N){const O=this.sliceSerialize(N),Y=this.stack[this.stack.length-2];Y.label=Sw(O),Y.identifier=Tt(O).toLowerCase()}function V(){const N=this.stack[this.stack.length-1],O=this.resume(),Y=this.stack[this.stack.length-1];if(this.data.inReference=!0,Y.type==="link"){const ee=N.children;Y.children=ee}else Y.alt=O}function v(){const N=this.resume(),O=this.stack[this.stack.length-1];O.url=N}function B(){const N=this.resume(),O=this.stack[this.stack.length-1];O.title=N}function $(){this.data.inReference=void 0}function w(){this.data.referenceType="collapsed"}function ie(N){const O=this.resume(),Y=this.stack[this.stack.length-1];Y.label=O,Y.identifier=Tt(this.sliceSerialize(N)).toLowerCase(),this.data.referenceType="full"}function ke(N){this.data.characterReferenceType=N.type}function le(N){const O=this.sliceSerialize(N),Y=this.data.characterReferenceType;let ee;Y?(ee=xh(O,Y==="characterReferenceMarkerNumeric"?10:16),this.data.characterReferenceType=void 0):ee=Xu(O);const ue=this.stack[this.stack.length-1];ue.value+=ee}function ct(N){const O=this.stack.pop();O.position.end=rn(N.end)}function q(N){k.call(this,N);const O=this.stack[this.stack.length-1];O.url=this.sliceSerialize(N)}function L(N){k.call(this,N);const O=this.stack[this.stack.length-1];O.url="mailto:"+this.sliceSerialize(N)}function M(){return{type:"blockquote",children:[]}}function H(){return{type:"code",lang:null,meta:null,value:""}}function ze(){return{type:"inlineCode",value:""}}function _e(){return{type:"definition",identifier:"",label:null,title:null,url:""}}function We(){return{type:"emphasis",children:[]}}function At(){return{type:"heading",depth:0,children:[]}}function Tn(){return{type:"break"}}function Ln(){return{type:"html",value:""}}function Cg(){return{type:"image",title:null,url:"",alt:null}}function oc(){return{type:"link",title:null,url:"",children:[]}}function sc(N){return{type:"list",ordered:N.type==="listOrdered",start:null,spread:N._spread,children:[]}}function zg(N){return{type:"listItem",spread:N._spread,checked:null,children:[]}}function Eg(){return{type:"paragraph",children:[]}}function Pg(){return{type:"strong",children:[]}}function Ng(){return{type:"text",value:""}}function Tg(){return{type:"thematicBreak"}}}function rn(e){return{line:e.line,column:e.column,offset:e.offset}}function Nh(e,t){let n=-1;for(;++n<t.length;){const r=t[n];Array.isArray(r)?Nh(e,r):Pw(e,r)}}function Pw(e,t){let n;for(n in t)if(Ph.call(t,n))switch(n){case"canContainEols":{const r=t[n];r&&e[n].push(...r);break}case"transforms":{const r=t[n];r&&e[n].push(...r);break}case"enter":case"exit":{const r=t[n];r&&Object.assign(e[n],r);break}}}function $d(e,t){throw e?new Error("Cannot close `"+e.type+"` ("+di({start:e.start,end:e.end})+"): a different token (`"+t.type+"`, "+di({start:t.start,end:t.end})+") is open"):new Error("Cannot close document, a token (`"+t.type+"`, "+di({start:t.start,end:t.end})+") is still open")}function Nw(e){const t=this;t.parser=n;function n(r){return zw(r,{...t.data("settings"),...e,extensions:t.data("micromarkExtensions")||[],mdastExtensions:t.data("fromMarkdownExtensions")||[]})}}function Tw(e,t){const n={type:"element",tagName:"blockquote",properties:{},children:e.wrap(e.all(t),!0)};return e.patch(t,n),e.applyData(t,n)}function Lw(e,t){const n={type:"element",tagName:"br",properties:{},children:[]};return e.patch(t,n),[e.applyData(t,n),{type:"text",value:`
`}]}function Aw(e,t){const n=t.value?t.value+`
`:"",r={},i=t.lang?t.lang.split(/\s+/):[];i.length>0&&(r.className=["language-"+i[0]]);let a={type:"element",tagName:"code",properties:r,children:[{type:"text",value:n}]};return t.meta&&(a.data={meta:t.meta}),e.patch(t,a),a=e.applyData(t,a),a={type:"element",tagName:"pre",properties:{},children:[a]},e.patch(t,a),a}function Rw(e,t){const n={type:"element",tagName:"del",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function jw(e,t){const n={type:"element",tagName:"em",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Iw(e,t){const n=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",r=String(t.identifier).toUpperCase(),i=Dr(r.toLowerCase()),a=e.footnoteOrder.indexOf(r);let o,s=e.footnoteCounts.get(r);s===void 0?(s=0,e.footnoteOrder.push(r),o=e.footnoteOrder.length):o=a+1,s+=1,e.footnoteCounts.set(r,s);const l={type:"element",tagName:"a",properties:{href:"#"+n+"fn-"+i,id:n+"fnref-"+i+(s>1?"-"+s:""),dataFootnoteRef:!0,ariaDescribedBy:["footnote-label"]},children:[{type:"text",value:String(o)}]};e.patch(t,l);const u={type:"element",tagName:"sup",properties:{},children:[l]};return e.patch(t,u),e.applyData(t,u)}function Dw(e,t){const n={type:"element",tagName:"h"+t.depth,properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Fw(e,t){if(e.options.allowDangerousHtml){const n={type:"raw",value:t.value};return e.patch(t,n),e.applyData(t,n)}}function Th(e,t){const n=t.referenceType;let r="]";if(n==="collapsed"?r+="[]":n==="full"&&(r+="["+(t.label||t.identifier)+"]"),t.type==="imageReference")return[{type:"text",value:"!["+t.alt+r}];const i=e.all(t),a=i[0];a&&a.type==="text"?a.value="["+a.value:i.unshift({type:"text",value:"["});const o=i[i.length-1];return o&&o.type==="text"?o.value+=r:i.push({type:"text",value:r}),i}function Mw(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return Th(e,t);const i={src:Dr(r.url||""),alt:t.alt};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"img",properties:i,children:[]};return e.patch(t,a),e.applyData(t,a)}function Ow(e,t){const n={src:Dr(t.url)};t.alt!==null&&t.alt!==void 0&&(n.alt=t.alt),t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"img",properties:n,children:[]};return e.patch(t,r),e.applyData(t,r)}function qw(e,t){const n={type:"text",value:t.value.replace(/\r?\n|\r/g," ")};e.patch(t,n);const r={type:"element",tagName:"code",properties:{},children:[n]};return e.patch(t,r),e.applyData(t,r)}function Ww(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return Th(e,t);const i={href:Dr(r.url||"")};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"a",properties:i,children:e.all(t)};return e.patch(t,a),e.applyData(t,a)}function Bw(e,t){const n={href:Dr(t.url)};t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"a",properties:n,children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function Uw(e,t,n){const r=e.all(t),i=n?Vw(n):Lh(t),a={},o=[];if(typeof t.checked=="boolean"){const d=r[0];let c;d&&d.type==="element"&&d.tagName==="p"?c=d:(c={type:"element",tagName:"p",properties:{},children:[]},r.unshift(c)),c.children.length>0&&c.children.unshift({type:"text",value:" "}),c.children.unshift({type:"element",tagName:"input",properties:{type:"checkbox",checked:t.checked,disabled:!0},children:[]}),a.className=["task-list-item"]}let s=-1;for(;++s<r.length;){const d=r[s];(i||s!==0||d.type!=="element"||d.tagName!=="p")&&o.push({type:"text",value:`
`}),d.type==="element"&&d.tagName==="p"&&!i?o.push(...d.children):o.push(d)}const l=r[r.length-1];l&&(i||l.type!=="element"||l.tagName!=="p")&&o.push({type:"text",value:`
`});const u={type:"element",tagName:"li",properties:a,children:o};return e.patch(t,u),e.applyData(t,u)}function Vw(e){let t=!1;if(e.type==="list"){t=e.spread||!1;const n=e.children;let r=-1;for(;!t&&++r<n.length;)t=Lh(n[r])}return t}function Lh(e){const t=e.spread;return t??e.children.length>1}function Hw(e,t){const n={},r=e.all(t);let i=-1;for(typeof t.start=="number"&&t.start!==1&&(n.start=t.start);++i<r.length;){const o=r[i];if(o.type==="element"&&o.tagName==="li"&&o.properties&&Array.isArray(o.properties.className)&&o.properties.className.includes("task-list-item")){n.className=["contains-task-list"];break}}const a={type:"element",tagName:t.ordered?"ol":"ul",properties:n,children:e.wrap(r,!0)};return e.patch(t,a),e.applyData(t,a)}function $w(e,t){const n={type:"element",tagName:"p",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Qw(e,t){const n={type:"root",children:e.wrap(e.all(t))};return e.patch(t,n),e.applyData(t,n)}function Xw(e,t){const n={type:"element",tagName:"strong",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function Kw(e,t){const n=e.all(t),r=n.shift(),i=[];if(r){const o={type:"element",tagName:"thead",properties:{},children:e.wrap([r],!0)};e.patch(t.children[0],o),i.push(o)}if(n.length>0){const o={type:"element",tagName:"tbody",properties:{},children:e.wrap(n,!0)},s=Uu(t.children[1]),l=ch(t.children[t.children.length-1]);s&&l&&(o.position={start:s,end:l}),i.push(o)}const a={type:"element",tagName:"table",properties:{},children:e.wrap(i,!0)};return e.patch(t,a),e.applyData(t,a)}function Yw(e,t,n){const r=n?n.children:void 0,a=(r?r.indexOf(t):1)===0?"th":"td",o=n&&n.type==="table"?n.align:void 0,s=o?o.length:t.children.length;let l=-1;const u=[];for(;++l<s;){const c=t.children[l],p={},f=o?o[l]:void 0;f&&(p.align=f);let y={type:"element",tagName:a,properties:p,children:[]};c&&(y.children=e.all(c),e.patch(c,y),y=e.applyData(c,y)),u.push(y)}const d={type:"element",tagName:"tr",properties:{},children:e.wrap(u,!0)};return e.patch(t,d),e.applyData(t,d)}function Gw(e,t){const n={type:"element",tagName:"td",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}const Qd=9,Xd=32;function Zw(e){const t=String(e),n=/\r?\n|\r/g;let r=n.exec(t),i=0;const a=[];for(;r;)a.push(Kd(t.slice(i,r.index),i>0,!0),r[0]),i=r.index+r[0].length,r=n.exec(t);return a.push(Kd(t.slice(i),i>0,!1)),a.join("")}function Kd(e,t,n){let r=0,i=e.length;if(t){let a=e.codePointAt(r);for(;a===Qd||a===Xd;)r++,a=e.codePointAt(r)}if(n){let a=e.codePointAt(i-1);for(;a===Qd||a===Xd;)i--,a=e.codePointAt(i-1)}return i>r?e.slice(r,i):""}function Jw(e,t){const n={type:"text",value:Zw(String(t.value))};return e.patch(t,n),e.applyData(t,n)}function ek(e,t){const n={type:"element",tagName:"hr",properties:{},children:[]};return e.patch(t,n),e.applyData(t,n)}const tk={blockquote:Tw,break:Lw,code:Aw,delete:Rw,emphasis:jw,footnoteReference:Iw,heading:Dw,html:Fw,imageReference:Mw,image:Ow,inlineCode:qw,linkReference:Ww,link:Bw,listItem:Uw,list:Hw,paragraph:$w,root:Qw,strong:Xw,table:Kw,tableCell:Gw,tableRow:Yw,text:Jw,thematicBreak:ek,toml:ma,yaml:ma,definition:ma,footnoteDefinition:ma};function ma(){}const Ah=-1,Mo=0,fi=1,co=2,Yu=3,Gu=4,Zu=5,Ju=6,Rh=7,jh=8,Yd=typeof self=="object"?self:globalThis,nk=(e,t)=>{const n=(i,a)=>(e.set(a,i),i),r=i=>{if(e.has(i))return e.get(i);const[a,o]=t[i];switch(a){case Mo:case Ah:return n(o,i);case fi:{const s=n([],i);for(const l of o)s.push(r(l));return s}case co:{const s=n({},i);for(const[l,u]of o)s[r(l)]=r(u);return s}case Yu:return n(new Date(o),i);case Gu:{const{source:s,flags:l}=o;return n(new RegExp(s,l),i)}case Zu:{const s=n(new Map,i);for(const[l,u]of o)s.set(r(l),r(u));return s}case Ju:{const s=n(new Set,i);for(const l of o)s.add(r(l));return s}case Rh:{const{name:s,message:l}=o;return n(new Yd[s](l),i)}case jh:return n(BigInt(o),i);case"BigInt":return n(Object(BigInt(o)),i);case"ArrayBuffer":return n(new Uint8Array(o).buffer,o);case"DataView":{const{buffer:s}=new Uint8Array(o);return n(new DataView(s),o)}}return n(new Yd[a](o),i)};return r},Gd=e=>nk(new Map,e)(0),Jn="",{toString:rk}={},{keys:ik}=Object,Xr=e=>{const t=typeof e;if(t!=="object"||!e)return[Mo,t];const n=rk.call(e).slice(8,-1);switch(n){case"Array":return[fi,Jn];case"Object":return[co,Jn];case"Date":return[Yu,Jn];case"RegExp":return[Gu,Jn];case"Map":return[Zu,Jn];case"Set":return[Ju,Jn];case"DataView":return[fi,n]}return n.includes("Array")?[fi,n]:n.includes("Error")?[Rh,n]:[co,n]},ha=([e,t])=>e===Mo&&(t==="function"||t==="symbol"),ak=(e,t,n,r)=>{const i=(o,s)=>{const l=r.push(o)-1;return n.set(s,l),l},a=o=>{if(n.has(o))return n.get(o);let[s,l]=Xr(o);switch(s){case Mo:{let d=o;switch(l){case"bigint":s=jh,d=o.toString();break;case"function":case"symbol":if(e)throw new TypeError("unable to serialize "+l);d=null;break;case"undefined":return i([Ah],o)}return i([s,d],o)}case fi:{if(l){let p=o;return l==="DataView"?p=new Uint8Array(o.buffer):l==="ArrayBuffer"&&(p=new Uint8Array(o)),i([l,[...p]],o)}const d=[],c=i([s,d],o);for(const p of o)d.push(a(p));return c}case co:{if(l)switch(l){case"BigInt":return i([l,o.toString()],o);case"Boolean":case"Number":case"String":return i([l,o.valueOf()],o)}if(t&&"toJSON"in o)return a(o.toJSON());const d=[],c=i([s,d],o);for(const p of ik(o))(e||!ha(Xr(o[p])))&&d.push([a(p),a(o[p])]);return c}case Yu:return i([s,o.toISOString()],o);case Gu:{const{source:d,flags:c}=o;return i([s,{source:d,flags:c}],o)}case Zu:{const d=[],c=i([s,d],o);for(const[p,f]of o)(e||!(ha(Xr(p))||ha(Xr(f))))&&d.push([a(p),a(f)]);return c}case Ju:{const d=[],c=i([s,d],o);for(const p of o)(e||!ha(Xr(p)))&&d.push(a(p));return c}}const{message:u}=o;return i([s,{name:l,message:u}],o)};return a},Zd=(e,{json:t,lossy:n}={})=>{const r=[];return ak(!(t||n),!!t,new Map,r)(e),r},po=typeof structuredClone=="function"?(e,t)=>t&&("json"in t||"lossy"in t)?Gd(Zd(e,t)):structuredClone(e):(e,t)=>Gd(Zd(e,t));function ok(e,t){const n=[{type:"text",value:"↩"}];return t>1&&n.push({type:"element",tagName:"sup",properties:{},children:[{type:"text",value:String(t)}]}),n}function sk(e,t){return"Back to reference "+(e+1)+(t>1?"-"+t:"")}function lk(e){const t=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",n=e.options.footnoteBackContent||ok,r=e.options.footnoteBackLabel||sk,i=e.options.footnoteLabel||"Footnotes",a=e.options.footnoteLabelTagName||"h2",o=e.options.footnoteLabelProperties||{className:["sr-only"]},s=[];let l=-1;for(;++l<e.footnoteOrder.length;){const u=e.footnoteById.get(e.footnoteOrder[l]);if(!u)continue;const d=e.all(u),c=String(u.identifier).toUpperCase(),p=Dr(c.toLowerCase());let f=0;const y=[],_=e.footnoteCounts.get(c);for(;_!==void 0&&++f<=_;){y.length>0&&y.push({type:"text",value:" "});let h=typeof n=="string"?n:n(l,f);typeof h=="string"&&(h={type:"text",value:h}),y.push({type:"element",tagName:"a",properties:{href:"#"+t+"fnref-"+p+(f>1?"-"+f:""),dataFootnoteBackref:"",ariaLabel:typeof r=="string"?r:r(l,f),className:["data-footnote-backref"]},children:Array.isArray(h)?h:[h]})}const S=d[d.length-1];if(S&&S.type==="element"&&S.tagName==="p"){const h=S.children[S.children.length-1];h&&h.type==="text"?h.value+=" ":S.children.push({type:"text",value:" "}),S.children.push(...y)}else d.push(...y);const m={type:"element",tagName:"li",properties:{id:t+"fn-"+p},children:e.wrap(d,!0)};e.patch(u,m),s.push(m)}if(s.length!==0)return{type:"element",tagName:"section",properties:{dataFootnotes:!0,className:["footnotes"]},children:[{type:"element",tagName:a,properties:{...po(o),id:"footnote-label"},children:[{type:"text",value:i}]},{type:"text",value:`
`},{type:"element",tagName:"ol",properties:{},children:e.wrap(s,!0)},{type:"text",value:`
`}]}}const Oo=function(e){if(e==null)return pk;if(typeof e=="function")return qo(e);if(typeof e=="object")return Array.isArray(e)?uk(e):ck(e);if(typeof e=="string")return dk(e);throw new Error("Expected function, string, or object as test")};function uk(e){const t=[];let n=-1;for(;++n<e.length;)t[n]=Oo(e[n]);return qo(r);function r(...i){let a=-1;for(;++a<t.length;)if(t[a].apply(this,i))return!0;return!1}}function ck(e){const t=e;return qo(n);function n(r){const i=r;let a;for(a in e)if(i[a]!==t[a])return!1;return!0}}function dk(e){return qo(t);function t(n){return n&&n.type===e}}function qo(e){return t;function t(n,r,i){return!!(fk(n)&&e.call(this,n,typeof r=="number"?r:void 0,i||void 0))}}function pk(){return!0}function fk(e){return e!==null&&typeof e=="object"&&"type"in e}const Ih=[],mk=!0,ql=!1,hk="skip";function Dh(e,t,n,r){let i;typeof t=="function"&&typeof n!="function"?(r=n,n=t):i=t;const a=Oo(i),o=r?-1:1;s(e,void 0,[])();function s(l,u,d){const c=l&&typeof l=="object"?l:{};if(typeof c.type=="string"){const f=typeof c.tagName=="string"?c.tagName:typeof c.name=="string"?c.name:void 0;Object.defineProperty(p,"name",{value:"node ("+(l.type+(f?"<"+f+">":""))+")"})}return p;function p(){let f=Ih,y,_,S;if((!t||a(l,u,d[d.length-1]||void 0))&&(f=gk(n(l,d)),f[0]===ql))return f;if("children"in l&&l.children){const m=l;if(m.children&&f[0]!==hk)for(_=(r?m.children.length:-1)+o,S=d.concat(m);_>-1&&_<m.children.length;){const h=m.children[_];if(y=s(h,_,S)(),y[0]===ql)return y;_=typeof y[1]=="number"?y[1]:_+o}}return f}}}function gk(e){return Array.isArray(e)?e:typeof e=="number"?[mk,e]:e==null?Ih:[e]}function ec(e,t,n,r){let i,a,o;typeof t=="function"&&typeof n!="function"?(a=void 0,o=t,i=n):(a=t,o=n,i=r),Dh(e,a,s,i);function s(l,u){const d=u[u.length-1],c=d?d.children.indexOf(l):void 0;return o(l,c,d)}}const Wl={}.hasOwnProperty,yk={};function xk(e,t){const n=t||yk,r=new Map,i=new Map,a=new Map,o={...tk,...n.handlers},s={all:u,applyData:vk,definitionById:r,footnoteById:i,footnoteCounts:a,footnoteOrder:[],handlers:o,one:l,options:n,patch:_k,wrap:kk};return ec(e,function(d){if(d.type==="definition"||d.type==="footnoteDefinition"){const c=d.type==="definition"?r:i,p=String(d.identifier).toUpperCase();c.has(p)||c.set(p,d)}}),s;function l(d,c){const p=d.type,f=s.handlers[p];if(Wl.call(s.handlers,p)&&f)return f(s,d,c);if(s.options.passThrough&&s.options.passThrough.includes(p)){if("children"in d){const{children:_,...S}=d,m=po(S);return m.children=s.all(d),m}return po(d)}return(s.options.unknownHandler||wk)(s,d,c)}function u(d){const c=[];if("children"in d){const p=d.children;let f=-1;for(;++f<p.length;){const y=s.one(p[f],d);if(y){if(f&&p[f-1].type==="break"&&(!Array.isArray(y)&&y.type==="text"&&(y.value=Jd(y.value)),!Array.isArray(y)&&y.type==="element")){const _=y.children[0];_&&_.type==="text"&&(_.value=Jd(_.value))}Array.isArray(y)?c.push(...y):c.push(y)}}}return c}}function _k(e,t){e.position&&(t.position=s_(e))}function vk(e,t){let n=t;if(e&&e.data){const r=e.data.hName,i=e.data.hChildren,a=e.data.hProperties;if(typeof r=="string")if(n.type==="element")n.tagName=r;else{const o="children"in n?n.children:[n];n={type:"element",tagName:r,properties:{},children:o}}n.type==="element"&&a&&Object.assign(n.properties,po(a)),"children"in n&&n.children&&i!==null&&i!==void 0&&(n.children=i)}return n}function wk(e,t){const n=t.data||{},r="value"in t&&!(Wl.call(n,"hProperties")||Wl.call(n,"hChildren"))?{type:"text",value:t.value}:{type:"element",tagName:"div",properties:{},children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function kk(e,t){const n=[];let r=-1;for(t&&n.push({type:"text",value:`
`});++r<e.length;)r&&n.push({type:"text",value:`
`}),n.push(e[r]);return t&&e.length>0&&n.push({type:"text",value:`
`}),n}function Jd(e){let t=0,n=e.charCodeAt(t);for(;n===9||n===32;)t++,n=e.charCodeAt(t);return e.slice(t)}function ep(e,t){const n=xk(e,t),r=n.one(e,void 0),i=lk(n),a=Array.isArray(r)?{type:"root",children:r}:r||{type:"root",children:[]};return i&&a.children.push({type:"text",value:`
`},i),a}function bk(e,t){return e&&"run"in e?async function(n,r){const i=ep(n,{file:r,...t});await e.run(i,r)}:function(n,r){return ep(n,{file:r,...e||t})}}function tp(e){if(e)throw e}var Ia=Object.prototype.hasOwnProperty,Fh=Object.prototype.toString,np=Object.defineProperty,rp=Object.getOwnPropertyDescriptor,ip=function(t){return typeof Array.isArray=="function"?Array.isArray(t):Fh.call(t)==="[object Array]"},ap=function(t){if(!t||Fh.call(t)!=="[object Object]")return!1;var n=Ia.call(t,"constructor"),r=t.constructor&&t.constructor.prototype&&Ia.call(t.constructor.prototype,"isPrototypeOf");if(t.constructor&&!n&&!r)return!1;var i;for(i in t);return typeof i>"u"||Ia.call(t,i)},op=function(t,n){np&&n.name==="__proto__"?np(t,n.name,{enumerable:!0,configurable:!0,value:n.newValue,writable:!0}):t[n.name]=n.newValue},sp=function(t,n){if(n==="__proto__")if(Ia.call(t,n)){if(rp)return rp(t,n).value}else return;return t[n]},Sk=function e(){var t,n,r,i,a,o,s=arguments[0],l=1,u=arguments.length,d=!1;for(typeof s=="boolean"&&(d=s,s=arguments[1]||{},l=2),(s==null||typeof s!="object"&&typeof s!="function")&&(s={});l<u;++l)if(t=arguments[l],t!=null)for(n in t)r=sp(s,n),i=sp(t,n),s!==i&&(d&&i&&(ap(i)||(a=ip(i)))?(a?(a=!1,o=r&&ip(r)?r:[]):o=r&&ap(r)?r:{},op(s,{name:n,newValue:e(d,o,i)})):typeof i<"u"&&op(s,{name:n,newValue:i}));return s};const vs=ho(Sk);function Bl(e){if(typeof e!="object"||e===null)return!1;const t=Object.getPrototypeOf(e);return(t===null||t===Object.prototype||Object.getPrototypeOf(t)===null)&&!(Symbol.toStringTag in e)&&!(Symbol.iterator in e)}function Ck(){const e=[],t={run:n,use:r};return t;function n(...i){let a=-1;const o=i.pop();if(typeof o!="function")throw new TypeError("Expected function as last argument, not "+o);s(null,...i);function s(l,...u){const d=e[++a];let c=-1;if(l){o(l);return}for(;++c<i.length;)(u[c]===null||u[c]===void 0)&&(u[c]=i[c]);i=u,d?zk(d,s)(...u):o(null,...u)}}function r(i){if(typeof i!="function")throw new TypeError("Expected `middelware` to be a function, not "+i);return e.push(i),t}}function zk(e,t){let n;return r;function r(...o){const s=e.length>o.length;let l;s&&o.push(i);try{l=e.apply(this,o)}catch(u){const d=u;if(s&&n)throw d;return i(d)}s||(l&&l.then&&typeof l.then=="function"?l.then(a,i):l instanceof Error?i(l):a(l))}function i(o,...s){n||(n=!0,t(o,...s))}function a(o){i(null,o)}}const It={basename:Ek,dirname:Pk,extname:Nk,join:Tk,sep:"/"};function Ek(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');Ki(e);let n=0,r=-1,i=e.length,a;if(t===void 0||t.length===0||t.length>e.length){for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else r<0&&(a=!0,r=i+1);return r<0?"":e.slice(n,r)}if(t===e)return"";let o=-1,s=t.length-1;for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else o<0&&(a=!0,o=i+1),s>-1&&(e.codePointAt(i)===t.codePointAt(s--)?s<0&&(r=i):(s=-1,r=o));return n===r?r=o:r<0&&(r=e.length),e.slice(n,r)}function Pk(e){if(Ki(e),e.length===0)return".";let t=-1,n=e.length,r;for(;--n;)if(e.codePointAt(n)===47){if(r){t=n;break}}else r||(r=!0);return t<0?e.codePointAt(0)===47?"/":".":t===1&&e.codePointAt(0)===47?"//":e.slice(0,t)}function Nk(e){Ki(e);let t=e.length,n=-1,r=0,i=-1,a=0,o;for(;t--;){const s=e.codePointAt(t);if(s===47){if(o){r=t+1;break}continue}n<0&&(o=!0,n=t+1),s===46?i<0?i=t:a!==1&&(a=1):i>-1&&(a=-1)}return i<0||n<0||a===0||a===1&&i===n-1&&i===r+1?"":e.slice(i,n)}function Tk(...e){let t=-1,n;for(;++t<e.length;)Ki(e[t]),e[t]&&(n=n===void 0?e[t]:n+"/"+e[t]);return n===void 0?".":Lk(n)}function Lk(e){Ki(e);const t=e.codePointAt(0)===47;let n=Ak(e,!t);return n.length===0&&!t&&(n="."),n.length>0&&e.codePointAt(e.length-1)===47&&(n+="/"),t?"/"+n:n}function Ak(e,t){let n="",r=0,i=-1,a=0,o=-1,s,l;for(;++o<=e.length;){if(o<e.length)s=e.codePointAt(o);else{if(s===47)break;s=47}if(s===47){if(!(i===o-1||a===1))if(i!==o-1&&a===2){if(n.length<2||r!==2||n.codePointAt(n.length-1)!==46||n.codePointAt(n.length-2)!==46){if(n.length>2){if(l=n.lastIndexOf("/"),l!==n.length-1){l<0?(n="",r=0):(n=n.slice(0,l),r=n.length-1-n.lastIndexOf("/")),i=o,a=0;continue}}else if(n.length>0){n="",r=0,i=o,a=0;continue}}t&&(n=n.length>0?n+"/..":"..",r=2)}else n.length>0?n+="/"+e.slice(i+1,o):n=e.slice(i+1,o),r=o-i-1;i=o,a=0}else s===46&&a>-1?a++:a=-1}return n}function Ki(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}const Rk={cwd:jk};function jk(){return"/"}function Ul(e){return!!(e!==null&&typeof e=="object"&&"href"in e&&e.href&&"protocol"in e&&e.protocol&&e.auth===void 0)}function Ik(e){if(typeof e=="string")e=new URL(e);else if(!Ul(e)){const t=new TypeError('The "path" argument must be of type string or an instance of URL. Received `'+e+"`");throw t.code="ERR_INVALID_ARG_TYPE",t}if(e.protocol!=="file:"){const t=new TypeError("The URL must be of scheme file");throw t.code="ERR_INVALID_URL_SCHEME",t}return Dk(e)}function Dk(e){if(e.hostname!==""){const r=new TypeError('File URL host must be "localhost" or empty on darwin');throw r.code="ERR_INVALID_FILE_URL_HOST",r}const t=e.pathname;let n=-1;for(;++n<t.length;)if(t.codePointAt(n)===37&&t.codePointAt(n+1)===50){const r=t.codePointAt(n+2);if(r===70||r===102){const i=new TypeError("File URL path must not include encoded / characters");throw i.code="ERR_INVALID_FILE_URL_PATH",i}}return decodeURIComponent(t)}const ws=["history","path","basename","stem","extname","dirname"];class Mh{constructor(t){let n;t?Ul(t)?n={path:t}:typeof t=="string"||Fk(t)?n={value:t}:n=t:n={},this.cwd="cwd"in n?"":Rk.cwd(),this.data={},this.history=[],this.messages=[],this.value,this.map,this.result,this.stored;let r=-1;for(;++r<ws.length;){const a=ws[r];a in n&&n[a]!==void 0&&n[a]!==null&&(this[a]=a==="history"?[...n[a]]:n[a])}let i;for(i in n)ws.includes(i)||(this[i]=n[i])}get basename(){return typeof this.path=="string"?It.basename(this.path):void 0}set basename(t){bs(t,"basename"),ks(t,"basename"),this.path=It.join(this.dirname||"",t)}get dirname(){return typeof this.path=="string"?It.dirname(this.path):void 0}set dirname(t){lp(this.basename,"dirname"),this.path=It.join(t||"",this.basename)}get extname(){return typeof this.path=="string"?It.extname(this.path):void 0}set extname(t){if(ks(t,"extname"),lp(this.dirname,"extname"),t){if(t.codePointAt(0)!==46)throw new Error("`extname` must start with `.`");if(t.includes(".",1))throw new Error("`extname` cannot contain multiple dots")}this.path=It.join(this.dirname,this.stem+(t||""))}get path(){return this.history[this.history.length-1]}set path(t){Ul(t)&&(t=Ik(t)),bs(t,"path"),this.path!==t&&this.history.push(t)}get stem(){return typeof this.path=="string"?It.basename(this.path,this.extname):void 0}set stem(t){bs(t,"stem"),ks(t,"stem"),this.path=It.join(this.dirname||"",t+(this.extname||""))}fail(t,n,r){const i=this.message(t,n,r);throw i.fatal=!0,i}info(t,n,r){const i=this.message(t,n,r);return i.fatal=void 0,i}message(t,n,r){const i=new qe(t,n,r);return this.path&&(i.name=this.path+":"+i.name,i.file=this.path),i.fatal=!1,this.messages.push(i),i}toString(t){return this.value===void 0?"":typeof this.value=="string"?this.value:new TextDecoder(t||void 0).decode(this.value)}}function ks(e,t){if(e&&e.includes(It.sep))throw new Error("`"+t+"` cannot be a path: did not expect `"+It.sep+"`")}function bs(e,t){if(!e)throw new Error("`"+t+"` cannot be empty")}function lp(e,t){if(!e)throw new Error("Setting `"+t+"` requires `path` to be set too")}function Fk(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const Mk=function(e){const r=this.constructor.prototype,i=r[e],a=function(){return i.apply(a,arguments)};return Object.setPrototypeOf(a,r),a},Ok={}.hasOwnProperty;class tc extends Mk{constructor(){super("copy"),this.Compiler=void 0,this.Parser=void 0,this.attachers=[],this.compiler=void 0,this.freezeIndex=-1,this.frozen=void 0,this.namespace={},this.parser=void 0,this.transformers=Ck()}copy(){const t=new tc;let n=-1;for(;++n<this.attachers.length;){const r=this.attachers[n];t.use(...r)}return t.data(vs(!0,{},this.namespace)),t}data(t,n){return typeof t=="string"?arguments.length===2?(zs("data",this.frozen),this.namespace[t]=n,this):Ok.call(this.namespace,t)&&this.namespace[t]||void 0:t?(zs("data",this.frozen),this.namespace=t,this):this.namespace}freeze(){if(this.frozen)return this;const t=this;for(;++this.freezeIndex<this.attachers.length;){const[n,...r]=this.attachers[this.freezeIndex];if(r[0]===!1)continue;r[0]===!0&&(r[0]=void 0);const i=n.call(t,...r);typeof i=="function"&&this.transformers.use(i)}return this.frozen=!0,this.freezeIndex=Number.POSITIVE_INFINITY,this}parse(t){this.freeze();const n=ga(t),r=this.parser||this.Parser;return Ss("parse",r),r(String(n),n)}process(t,n){const r=this;return this.freeze(),Ss("process",this.parser||this.Parser),Cs("process",this.compiler||this.Compiler),n?i(void 0,n):new Promise(i);function i(a,o){const s=ga(t),l=r.parse(s);r.run(l,s,function(d,c,p){if(d||!c||!p)return u(d);const f=c,y=r.stringify(f,p);Bk(y)?p.value=y:p.result=y,u(d,p)});function u(d,c){d||!c?o(d):a?a(c):n(void 0,c)}}}processSync(t){let n=!1,r;return this.freeze(),Ss("processSync",this.parser||this.Parser),Cs("processSync",this.compiler||this.Compiler),this.process(t,i),cp("processSync","process",n),r;function i(a,o){n=!0,tp(a),r=o}}run(t,n,r){up(t),this.freeze();const i=this.transformers;return!r&&typeof n=="function"&&(r=n,n=void 0),r?a(void 0,r):new Promise(a);function a(o,s){const l=ga(n);i.run(t,l,u);function u(d,c,p){const f=c||t;d?s(d):o?o(f):r(void 0,f,p)}}}runSync(t,n){let r=!1,i;return this.run(t,n,a),cp("runSync","run",r),i;function a(o,s){tp(o),i=s,r=!0}}stringify(t,n){this.freeze();const r=ga(n),i=this.compiler||this.Compiler;return Cs("stringify",i),up(t),i(t,r)}use(t,...n){const r=this.attachers,i=this.namespace;if(zs("use",this.frozen),t!=null)if(typeof t=="function")l(t,n);else if(typeof t=="object")Array.isArray(t)?s(t):o(t);else throw new TypeError("Expected usable value, not `"+t+"`");return this;function a(u){if(typeof u=="function")l(u,[]);else if(typeof u=="object")if(Array.isArray(u)){const[d,...c]=u;l(d,c)}else o(u);else throw new TypeError("Expected usable value, not `"+u+"`")}function o(u){if(!("plugins"in u)&&!("settings"in u))throw new Error("Expected usable value but received an empty preset, which is probably a mistake: presets typically come with `plugins` and sometimes with `settings`, but this has neither");s(u.plugins),u.settings&&(i.settings=vs(!0,i.settings,u.settings))}function s(u){let d=-1;if(u!=null)if(Array.isArray(u))for(;++d<u.length;){const c=u[d];a(c)}else throw new TypeError("Expected a list of plugins, not `"+u+"`")}function l(u,d){let c=-1,p=-1;for(;++c<r.length;)if(r[c][0]===u){p=c;break}if(p===-1)r.push([u,...d]);else if(d.length>0){let[f,...y]=d;const _=r[p][1];Bl(_)&&Bl(f)&&(f=vs(!0,_,f)),r[p]=[u,f,...y]}}}}const qk=new tc().freeze();function Ss(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `parser`")}function Cs(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `compiler`")}function zs(e,t){if(t)throw new Error("Cannot call `"+e+"` on a frozen processor.\nCreate a new processor first, by calling it: use `processor()` instead of `processor`.")}function up(e){if(!Bl(e)||typeof e.type!="string")throw new TypeError("Expected node, got `"+e+"`")}function cp(e,t,n){if(!n)throw new Error("`"+e+"` finished async. Use `"+t+"` instead")}function ga(e){return Wk(e)?e:new Mh(e)}function Wk(e){return!!(e&&typeof e=="object"&&"message"in e&&"messages"in e)}function Bk(e){return typeof e=="string"||Uk(e)}function Uk(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const Vk="https://github.com/remarkjs/react-markdown/blob/main/changelog.md",dp=[],pp={allowDangerousHtml:!0},Hk=/^(https?|ircs?|mailto|xmpp)$/i,$k=[{from:"astPlugins",id:"remove-buggy-html-in-markdown-parser"},{from:"allowDangerousHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"allowNode",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowElement"},{from:"allowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowedElements"},{from:"disallowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"disallowedElements"},{from:"escapeHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"includeElementIndex",id:"#remove-includeelementindex"},{from:"includeNodeIndex",id:"change-includenodeindex-to-includeelementindex"},{from:"linkTarget",id:"remove-linktarget"},{from:"plugins",id:"change-plugins-to-remarkplugins",to:"remarkPlugins"},{from:"rawSourcePos",id:"#remove-rawsourcepos"},{from:"renderers",id:"change-renderers-to-components",to:"components"},{from:"source",id:"change-source-to-children",to:"children"},{from:"sourcePos",id:"#remove-sourcepos"},{from:"transformImageUri",id:"#add-urltransform",to:"urlTransform"},{from:"transformLinkUri",id:"#add-urltransform",to:"urlTransform"}];function Oh(e){const t=Qk(e),n=Xk(e);return Kk(t.runSync(t.parse(n),n),e)}function Qk(e){const t=e.rehypePlugins||dp,n=e.remarkPlugins||dp,r=e.remarkRehypeOptions?{...e.remarkRehypeOptions,...pp}:pp;return qk().use(Nw).use(n).use(bk,r).use(t)}function Xk(e){const t=e.children||"",n=new Mh;return typeof t=="string"&&(n.value=t),n}function Kk(e,t){const n=t.allowedElements,r=t.allowElement,i=t.components,a=t.disallowedElements,o=t.skipHtml,s=t.unwrapDisallowed,l=t.urlTransform||Yk;for(const d of $k)Object.hasOwn(t,d.from)&&(""+d.from+(d.to?"use `"+d.to+"` instead":"remove it")+Vk+d.id,void 0);return t.className&&(e={type:"element",tagName:"div",properties:{className:t.className},children:e.type==="root"?e.children:[e]}),ec(e,u),p_(e,{Fragment:x.Fragment,components:i,ignoreInvalidStyle:!0,jsx:x.jsx,jsxs:x.jsxs,passKeys:!0,passNode:!0});function u(d,c,p){if(d.type==="raw"&&p&&typeof c=="number")return o?p.children.splice(c,1):p.children[c]={type:"text",value:d.value},c;if(d.type==="element"){let f;for(f in ys)if(Object.hasOwn(ys,f)&&Object.hasOwn(d.properties,f)){const y=d.properties[f],_=ys[f];(_===null||_.includes(d.tagName))&&(d.properties[f]=l(String(y||""),f,d))}}if(d.type==="element"){let f=n?!n.includes(d.tagName):a?a.includes(d.tagName):!1;if(!f&&r&&typeof c=="number"&&(f=!r(d,c,p)),f&&p&&typeof c=="number")return s&&d.children?p.children.splice(c,1,...d.children):p.children.splice(c,1),c}}}function Yk(e){const t=e.indexOf(":"),n=e.indexOf("?"),r=e.indexOf("#"),i=e.indexOf("/");return t===-1||i!==-1&&t>i||n!==-1&&t>n||r!==-1&&t>r||Hk.test(e.slice(0,t))?e:""}const Gk=[{id:"numpy-array-sum",title:"Array Sum",section:"python-basics",difficulty:"easy",description:`
## Array Sum

Given a NumPy array of numbers, implement a function that returns the sum of all elements.

### Function Signature
\`\`\`python
def array_sum(arr: np.ndarray) -> float:
\`\`\`

### Constraints
- Array length: 1 <= n <= 1000
- Element values: -10^6 <= arr[i] <= 10^6
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"15",explanation:"1 + 2 + 3 + 4 + 5 = 15"},{input:"np.array([-1, 0, 1])",output:"0",explanation:"-1 + 0 + 1 = 0"}],starterCode:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.

    Args:
        arr: A NumPy array of numbers

    Returns:
        The sum of all elements
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic positive numbers",input:"[1, 2, 3, 4, 5]",expected:"15",hidden:!1},{id:"2",description:"With negatives",input:"[-1, 0, 1]",expected:"0",hidden:!1},{id:"3",description:"Single element",input:"[42]",expected:"42",hidden:!1},{id:"4",description:"Larger array",input:"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",expected:"55",hidden:!0}],hints:["NumPy arrays have built-in methods for common operations.","Try using np.sum() or the .sum() method on the array."],solution:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.
    """
    return np.sum(arr)
`},{id:"numpy-matrix-multiply",title:"Matrix Multiplication",section:"python-basics",difficulty:"medium",description:`
## Matrix Multiplication

Implement matrix multiplication between two 2D NumPy arrays.

### Function Signature
\`\`\`python
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- A has shape (m, n)
- B has shape (n, p)
- Result has shape (m, p)
- 1 <= m, n, p <= 100

### Note
You should use NumPy's built-in functions, not manual loops.
    `,examples:[{input:"A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]",output:"[[19, 22], [43, 50]]",explanation:"Standard matrix multiplication"}],starterCode:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.

    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)

    Returns:
        Result matrix of shape (m, p)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2x2 matrices",input:"([[1, 2], [3, 4]], [[5, 6], [7, 8]])",expected:"[[19, 22], [43, 50]]",hidden:!1},{id:"2",description:"Identity multiplication",input:"([[1, 0], [0, 1]], [[5, 6], [7, 8]])",expected:"[[5, 6], [7, 8]]",hidden:!1},{id:"3",description:"Different dimensions",input:"([[1, 2, 3]], [[1], [2], [3]])",expected:"[[14]]",hidden:!0}],hints:["NumPy provides several ways to do matrix multiplication.","Try np.dot(), np.matmul(), or the @ operator."],solution:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.
    """
    return np.dot(A, B)
    # Alternative: return A @ B
    # Alternative: return np.matmul(A, B)
`},{id:"numpy-broadcast-add",title:"Broadcasting Addition",section:"python-basics",difficulty:"medium",description:`
## Broadcasting Addition

Given a 2D matrix and a 1D vector, add the vector to each row of the matrix using broadcasting.

### Function Signature
\`\`\`python
def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Matrix has shape (m, n)
- Vector has shape (n,)
- 1 <= m, n <= 100

### Note
Broadcasting is a powerful NumPy feature that allows operations between arrays of different shapes.
    `,examples:[{input:"matrix = [[1, 2, 3], [4, 5, 6]], vector = [10, 20, 30]",output:"[[11, 22, 33], [14, 25, 36]]",explanation:"Vector is added to each row"}],starterCode:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.

    Args:
        matrix: 2D array of shape (m, n)
        vector: 1D array of shape (n,)

    Returns:
        Result matrix of shape (m, n)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic broadcasting",input:"([[1, 2, 3], [4, 5, 6]], [10, 20, 30])",expected:"[[11, 22, 33], [14, 25, 36]]",hidden:!1},{id:"2",description:"With zeros",input:"([[1, 2], [3, 4]], [0, 0])",expected:"[[1, 2], [3, 4]]",hidden:!1},{id:"3",description:"Negative values",input:"([[1, 2], [3, 4]], [-1, -2])",expected:"[[0, 0], [2, 2]]",hidden:!0}],hints:["NumPy automatically broadcasts the vector across rows.","Simply use the + operator - NumPy handles the rest!"],solution:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.
    """
    return matrix + vector
`}],Zk=[{id:"normalize-features",title:"Normalize Features",section:"data-preprocessing",difficulty:"easy",description:`
## Normalize Features (Min-Max Scaling)

Implement min-max normalization to scale features to the range [0, 1].

### Formula
\`\`\`
X_normalized = (X - X_min) / (X_max - X_min)
\`\`\`

### Function Signature
\`\`\`python
def normalize(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array length: 2 <= n <= 1000
- All elements are finite numbers
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"[0.0, 0.25, 0.5, 0.75, 1.0]",explanation:"min=1, max=5, so (x-1)/(5-1) gives [0, 0.25, 0.5, 0.75, 1]"}],starterCode:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].

    Args:
        arr: Input array of numbers

    Returns:
        Normalized array with values in [0, 1]
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic normalization",input:"[1, 2, 3, 4, 5]",expected:"[0.0, 0.25, 0.5, 0.75, 1.0]",hidden:!1},{id:"2",description:"With negative values",input:"[-10, 0, 10]",expected:"[0.0, 0.5, 1.0]",hidden:!1},{id:"3",description:"Larger range",input:"[0, 50, 100]",expected:"[0.0, 0.5, 1.0]",hidden:!0}],hints:["Use np.min() and np.max() to find the range.","Apply the formula: (x - min) / (max - min)"],solution:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)
`},{id:"handle-missing-data",title:"Handle Missing Data",section:"data-preprocessing",difficulty:"medium",description:`
## Handle Missing Data

Replace NaN (missing) values in an array with the mean of non-NaN values.

### Function Signature
\`\`\`python
def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array contains at least one non-NaN value
- NaN values are represented as np.nan
    `,examples:[{input:"np.array([1.0, np.nan, 3.0, np.nan, 5.0])",output:"[1.0, 3.0, 3.0, 3.0, 5.0]",explanation:"Mean of [1, 3, 5] is 3, which replaces NaN values"}],starterCode:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.

    Args:
        arr: Input array that may contain NaN values

    Returns:
        Array with NaN values replaced by the mean
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic case",input:'[1.0, float("nan"), 3.0, float("nan"), 5.0]',expected:"[1.0, 3.0, 3.0, 3.0, 5.0]",hidden:!1},{id:"2",description:"Single NaN",input:'[2.0, 4.0, float("nan"), 6.0]',expected:"[2.0, 4.0, 4.0, 6.0]",hidden:!1},{id:"3",description:"NaN at start",input:'[float("nan"), 10.0, 20.0]',expected:"[15.0, 10.0, 20.0]",hidden:!0}],hints:["Use np.isnan() to find NaN values.","Use np.nanmean() to compute mean ignoring NaN values.","Use boolean indexing to replace values."],solution:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.
    """
    arr = arr.copy()  # Don't modify original
    mean_val = np.nanmean(arr)
    arr[np.isnan(arr)] = mean_val
    return arr
`},{id:"one-hot-encode",title:"One-Hot Encoding",section:"data-preprocessing",difficulty:"medium",description:`
## One-Hot Encoding

Convert an array of categorical integers to one-hot encoded format.

### Function Signature
\`\`\`python
def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
\`\`\`

### Example
For labels [0, 1, 2] with 3 classes:
\`\`\`
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
\`\`\`

### Constraints
- Labels are integers in range [0, num_classes - 1]
- 1 <= num_classes <= 100
    `,examples:[{input:"labels = [0, 1, 2, 1], num_classes = 3",output:"[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]",explanation:"Each label becomes a row with 1 at the label index"}],starterCode:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.

    Args:
        labels: 1D array of integer labels
        num_classes: Total number of classes

    Returns:
        2D array of shape (len(labels), num_classes)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic encoding",input:"([0, 1, 2], 3)",expected:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",hidden:!1},{id:"2",description:"Repeated labels",input:"([0, 0, 1, 1], 2)",expected:"[[1, 0], [1, 0], [0, 1], [0, 1]]",hidden:!1},{id:"3",description:"More classes than used",input:"([0, 2], 4)",expected:"[[1, 0, 0, 0], [0, 0, 1, 0]]",hidden:!0}],hints:["Create a zeros matrix of shape (num_samples, num_classes).","Use array indexing to set the appropriate positions to 1.","np.eye() can be useful for creating one-hot encodings."],solution:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.
    """
    # Method 1: Using np.eye
    return np.eye(num_classes)[labels].astype(int)

    # Method 2: Manual approach
    # one_hot = np.zeros((len(labels), num_classes), dtype=int)
    # one_hot[np.arange(len(labels)), labels] = 1
    # return one_hot
`}],Jk=[{id:"linear-regression-gd",title:"Linear Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Linear Regression with Gradient Descent

Implement simple linear regression using gradient descent optimization.

### Model
\`\`\`
y = w * x + b
\`\`\`

### Gradient Descent Updates
\`\`\`
w = w - learning_rate * dw
b = b - learning_rate * db
\`\`\`

Where:
- dw = (2/n) * sum((y_pred - y) * x)
- db = (2/n) * sum(y_pred - y)

### Function Signature
\`\`\`python
def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float, iterations: int) -> tuple:
\`\`\`

Returns (w, b) - the learned weight and bias.
    `,examples:[{input:"X = [1, 2, 3, 4], y = [2, 4, 6, 8], lr = 0.1, iters = 1000",output:"w ≈ 2.0, b ≈ 0.0",explanation:"The true relationship is y = 2x, so w should be close to 2"}],starterCode:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.

    Args:
        X: Input features (1D array)
        y: Target values (1D array)
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations

    Returns:
        Tuple of (weight, bias)
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    n = len(X)

    # Your gradient descent implementation here
    pass

    return (round(w, 2), round(b, 2))
`,testCases:[{id:"1",description:"Perfect linear relationship",input:"([1, 2, 3, 4], [2, 4, 6, 8], 0.1, 1000)",expected:"(2.0, 0.0)",hidden:!1},{id:"2",description:"With intercept",input:"([1, 2, 3, 4], [3, 5, 7, 9], 0.1, 1000)",expected:"(2.0, 1.0)",hidden:!1},{id:"3",description:"Different slope",input:"([0, 1, 2, 3], [1, 4, 7, 10], 0.1, 1000)",expected:"(3.0, 1.0)",hidden:!0}],hints:["First compute predictions: y_pred = w * X + b","Then compute gradients: dw = (2/n) * sum((y_pred - y) * X)","Update weights: w = w - learning_rate * dw"],solution:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.
    """
    w = 0.0
    b = 0.0
    n = len(X)

    for _ in range(iterations):
        # Forward pass
        y_pred = w * X + b

        # Compute gradients
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return (round(w, 2), round(b, 2))
`},{id:"logistic-regression",title:"Sigmoid Function",section:"supervised-learning",difficulty:"easy",description:`
## Sigmoid Function

Implement the sigmoid activation function used in logistic regression.

### Formula
\`\`\`
sigmoid(x) = 1 / (1 + exp(-x))
\`\`\`

### Properties
- Output is always between 0 and 1
- sigmoid(0) = 0.5
- Monotonically increasing

### Function Signature
\`\`\`python
def sigmoid(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([0, 1, -1])",output:"[0.5, 0.731059, 0.268941]",explanation:"sigmoid(0)=0.5, sigmoid(1)≈0.73, sigmoid(-1)≈0.27"}],starterCode:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with sigmoid applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zero input",input:"[0]",expected:"[0.5]",hidden:!1},{id:"2",description:"Positive and negative",input:"bool(np.allclose(sigmoid(np.array([-1, 0, 1])), [0.268941, 0.5, 0.731059], atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(np.allclose(sigmoid(np.array([-10, 10])), [4.5e-05, 0.999955], atol=1e-5))",expected:"True",hidden:!0}],hints:["Use np.exp() for the exponential function.","The formula is 1 / (1 + exp(-x))"],solution:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))
`},{id:"decision-tree-split",title:"Gini Impurity",section:"supervised-learning",difficulty:"medium",description:`
## Gini Impurity

Calculate the Gini impurity for a set of labels. This metric is used in decision trees to determine the best split.

### Formula
\`\`\`
Gini = 1 - sum(p_i^2)
\`\`\`

Where p_i is the proportion of class i in the set.

### Properties
- Gini = 0 means pure (all same class)
- Gini = 0.5 means maximum impurity (for binary classification with equal split)

### Function Signature
\`\`\`python
def gini_impurity(labels: np.ndarray) -> float:
\`\`\`
    `,examples:[{input:"np.array([0, 0, 0, 0])",output:"0.0",explanation:"All same class, pure node"},{input:"np.array([0, 0, 1, 1])",output:"0.5",explanation:"Equal split, maximum impurity for binary"}],starterCode:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.

    Args:
        labels: Array of class labels (integers)

    Returns:
        Gini impurity value between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Pure node",input:"[0, 0, 0, 0]",expected:"0.0",hidden:!1},{id:"2",description:"Maximum impurity",input:"[0, 0, 1, 1]",expected:"0.5",hidden:!1},{id:"3",description:"Unequal split",input:"[0, 0, 0, 1]",expected:"0.375",hidden:!0}],hints:["Count the occurrences of each class.","Calculate the proportion (probability) of each class.","Gini = 1 - sum of squared probabilities."],solution:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.
    """
    if len(labels) == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(labels, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(labels)

    # Gini = 1 - sum(p^2)
    return 1 - np.sum(probabilities ** 2)
`},{id:"logistic-regression-full",title:"Logistic Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Logistic Regression with Gradient Descent

Implement binary logistic regression from scratch.

### Model
\`\`\`
z = X @ w + b
y_pred = sigmoid(z) = 1 / (1 + exp(-z))
\`\`\`

### Loss (Binary Cross-Entropy)
\`\`\`
L = -1/m * sum(y * log(y_pred) + (1-y) * log(1-y_pred))
\`\`\`

### Gradients
\`\`\`
dw = 1/m * X.T @ (y_pred - y)
db = 1/m * sum(y_pred - y)
\`\`\`
    `,examples:[{input:"X (100, 2), y binary labels, 1000 iterations",output:"Trained weights and bias",explanation:"Learns decision boundary separating classes"}],starterCode:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    """
    Train logistic regression using gradient descent.

    Args:
        X: Features (m samples, n features)
        y: Binary labels (m,)
        learning_rate: Step size
        iterations: Number of iterations

    Returns:
        w: Learned weights (n,)
        b: Learned bias (scalar)
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    # Your code here
    pass

    return np.round(w, 4), round(b, 4)
`,testCases:[{id:"1",description:"Simple separable data",input:"(lambda r: bool(np.allclose(r[0], [6.0141, 6.0141], atol=1e-3) and np.isclose(r[1], -9.1984, atol=1e-3)))(logistic_regression(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]), 0.5, 1000))",expected:"True",hidden:!1}],hints:["Forward: z = X @ w + b, then y_pred = sigmoid(z)","Gradients: dw = (1/m) * X.T @ (y_pred - y)","Update: w = w - lr * dw, b = b - lr * db"],solution:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        # Forward pass
        z = X @ w + b
        y_pred = sigmoid(z)

        # Compute gradients
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return np.round(w, 4), round(b, 4)
`},{id:"binary-cross-entropy",title:"Binary Cross-Entropy Loss",section:"supervised-learning",difficulty:"easy",description:`
## Binary Cross-Entropy Loss

Implement the binary cross-entropy (log loss) function.

### Formula
\`\`\`
BCE = -1/m * sum(y * log(p) + (1-y) * log(1-p))
\`\`\`

Where:
- y: True labels (0 or 1)
- p: Predicted probabilities
- m: Number of samples

### Numerical Stability
Clip predictions to avoid log(0):
\`\`\`python
p = np.clip(p, 1e-15, 1 - 1e-15)
\`\`\`
    `,examples:[{input:"y = [1, 0, 1], p = [0.9, 0.1, 0.8]",output:"0.1446",explanation:"Low loss for confident correct predictions"}],starterCode:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities

    Returns:
        loss: Scalar BCE loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect predictions",input:"([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0])",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"([1, 0, 1], [0.9, 0.1, 0.8])",expected:"0.1446",hidden:!1}],hints:["Clip predictions for numerical stability","Apply the formula element-wise","Take the mean over all samples"],solution:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip for numerical stability
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # BCE formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return round(loss, 4)
`}],eb=[{id:"kmeans-clustering",title:"K-Means: Assign to Nearest Centroid",section:"unsupervised-learning",difficulty:"medium",description:`
## K-Means: Assign Points to Nearest Centroid

Implement the assignment step of K-means clustering: given data points and centroids, assign each point to its nearest centroid.

### Function Signature
\`\`\`python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data points of shape (n_samples, n_features)
- centroids: Cluster centers of shape (k, n_features)

### Returns
- Array of cluster assignments (integers 0 to k-1)

### Distance
Use Euclidean distance: sqrt(sum((x - c)^2))
    `,examples:[{input:"X = [[0, 0], [1, 1], [10, 10]], centroids = [[0, 0], [10, 10]]",output:"[0, 0, 1]",explanation:"First two points closer to centroid 0, last point closer to centroid 1"}],starterCode:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.

    Args:
        X: Data points of shape (n_samples, n_features)
        centroids: Cluster centers of shape (k, n_features)

    Returns:
        Array of cluster assignments (shape: n_samples)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple 2D case",input:"([[0, 0], [1, 1], [10, 10]], [[0, 0], [10, 10]])",expected:"[0, 0, 1]",hidden:!1},{id:"2",description:"Three clusters",input:"([[0, 0], [5, 5], [10, 10]], [[0, 0], [5, 5], [10, 10]])",expected:"[0, 1, 2]",hidden:!1},{id:"3",description:"Tiebreaker (first centroid wins)",input:"([[5, 5]], [[0, 0], [10, 10]])",expected:"[0]",hidden:!0}],hints:["Calculate distance from each point to each centroid.","Use np.linalg.norm() for Euclidean distance.","Use np.argmin() to find the closest centroid."],solution:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.
    """
    # Calculate distances from each point to each centroid
    # Using broadcasting: X[:, np.newaxis] has shape (n, 1, d)
    # centroids has shape (k, d)
    # Difference has shape (n, k, d)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    # Return index of minimum distance for each point
    return np.argmin(distances, axis=1)
`},{id:"pca-implementation",title:"PCA: Center Data",section:"unsupervised-learning",difficulty:"easy",description:`
## PCA: Center the Data

The first step in PCA is to center the data by subtracting the mean of each feature.

### Function Signature
\`\`\`python
def center_data(X: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data matrix of shape (n_samples, n_features)

### Returns
- Centered data where each column has mean 0

### Why Center?
Centering ensures PCA finds directions of maximum variance from the origin, not from an arbitrary point.
    `,examples:[{input:"X = [[1, 2], [3, 4], [5, 6]]",output:"[[-2, -2], [0, 0], [2, 2]]",explanation:"Column means are [3, 4], so we subtract them"}],starterCode:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Centered data matrix of same shape
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic centering",input:"[[1, 2], [3, 4], [5, 6]]",expected:"[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]",hidden:!1},{id:"2",description:"Already centered",input:"[[-1, -1], [0, 0], [1, 1]]",expected:"[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]",hidden:!1},{id:"3",description:"Single column",input:"[[10], [20], [30]]",expected:"[[-10.0], [0.0], [10.0]]",hidden:!0}],hints:["Calculate the mean of each column using np.mean() with axis=0.","Subtract the mean from the data (broadcasting handles this)."],solution:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.
    """
    mean = np.mean(X, axis=0)
    return X - mean
`}],tb=[{id:"perceptron",title:"ReLU Activation",section:"deep-learning",difficulty:"easy",description:`
## ReLU Activation Function

Implement the Rectified Linear Unit (ReLU) activation function, one of the most popular activation functions in deep learning.

### Formula
\`\`\`
ReLU(x) = max(0, x)
\`\`\`

### Properties
- Output is x if x > 0, else 0
- Helps with vanishing gradient problem
- Computationally efficient

### Function Signature
\`\`\`python
def relu(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([-2, -1, 0, 1, 2])",output:"[0, 0, 0, 1, 2]",explanation:"Negative values become 0, positive values unchanged"}],starterCode:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array with ReLU applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed values",input:"[-2, -1, 0, 1, 2]",expected:"[0, 0, 0, 1, 2]",hidden:!1},{id:"2",description:"All negative",input:"[-5, -3, -1]",expected:"[0, 0, 0]",hidden:!1},{id:"3",description:"All positive",input:"[1, 2, 3]",expected:"[1, 2, 3]",hidden:!0}],hints:["Use np.maximum() to compare with 0.","Alternatively, use boolean indexing or np.where()."],solution:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.
    """
    return np.maximum(0, x)
`},{id:"neural-network-forward",title:"Dense Layer Forward Pass",section:"deep-learning",difficulty:"medium",description:`
## Dense Layer Forward Pass

Implement the forward pass of a fully-connected (dense) neural network layer.

### Formula
\`\`\`
output = activation(X @ W + b)
\`\`\`

Where:
- X: Input of shape (batch_size, input_dim)
- W: Weights of shape (input_dim, output_dim)
- b: Bias of shape (output_dim,)

### Function Signature
\`\`\`python
def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
\`\`\`

Use ReLU as the activation function.
    `,examples:[{input:"X = [[1, 2]], W = [[1, 0], [0, 1]], b = [1, 1]",output:"[[2, 3]]",explanation:"X @ W = [[1, 2]], + b = [[2, 3]], ReLU keeps positive"}],starterCode:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.

    Args:
        X: Input of shape (batch_size, input_dim)
        W: Weights of shape (input_dim, output_dim)
        b: Bias of shape (output_dim,)

    Returns:
        Output of shape (batch_size, output_dim)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic forward pass",input:"([[1, 2]], [[1, 0], [0, 1]], [1, 1])",expected:"[[2, 3]]",hidden:!1},{id:"2",description:"With negative pre-activation",input:"([[1, 1]], [[1, -2], [-1, 1]], [0, 0])",expected:"[[0, 0]]",hidden:!1},{id:"3",description:"Batch of 2",input:"([[1, 0], [0, 1]], [[2, 0], [0, 2]], [0, 0])",expected:"[[2, 0], [0, 2]]",hidden:!0}],hints:["First compute the linear transformation: Z = X @ W + b","Then apply ReLU: output = max(0, Z)"],solution:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.
    """
    # Linear transformation
    Z = np.dot(X, W) + b
    # ReLU activation
    return np.maximum(0, Z)
`},{id:"backpropagation",title:"Softmax Function",section:"deep-learning",difficulty:"medium",description:`
## Softmax Function

Implement the softmax function, used to convert raw scores to probabilities in multi-class classification.

### Formula
\`\`\`
softmax(x)_i = exp(x_i) / sum(exp(x_j))
\`\`\`

### Properties
- Output sums to 1
- All outputs are positive
- Larger inputs get larger probabilities

### Function Signature
\`\`\`python
def softmax(x: np.ndarray) -> np.ndarray:
\`\`\`

**Note**: For numerical stability, subtract max(x) before computing exp.
    `,examples:[{input:"np.array([1, 2, 3])",output:"[0.090031, 0.244728, 0.665241]",explanation:"Higher values get higher probabilities, sum = 1"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input array (1D)

    Returns:
        Array of same shape with softmax probabilities
    """
    # Your code here
    # Hint: subtract max for numerical stability
    pass
`,testCases:[{id:"1",description:"Basic softmax",input:"bool(np.allclose(softmax(np.array([1, 2, 3])), [0.090031, 0.244728, 0.665241], atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Equal inputs",input:"bool(np.allclose(softmax(np.array([1, 1, 1])), [0.333333, 0.333333, 0.333333], atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(np.allclose(softmax(np.array([100, 101, 102])), [0.090031, 0.244728, 0.665241], atol=1e-5))",expected:"True",hidden:!0}],hints:["Subtract the max value for numerical stability.","Compute exp of each element.","Divide by the sum of all exp values."],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
`}],nb=[{id:"precision-recall-f1",title:"Precision and Recall",section:"model-evaluation",difficulty:"medium",description:`
## Precision and Recall

Calculate precision and recall from true labels and predictions.

### Formulas
\`\`\`
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
\`\`\`

Where:
- TP = True Positives (predicted 1, actual 1)
- FP = False Positives (predicted 1, actual 0)
- FN = False Negatives (predicted 0, actual 1)

### Function Signature
\`\`\`python
def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
\`\`\`

Returns (precision, recall) rounded to 4 decimal places.
    `,examples:[{input:"y_true = [1, 1, 0, 1, 0], y_pred = [1, 0, 0, 1, 1]",output:"(0.6667, 0.6667)",explanation:"TP=2, FP=1, FN=1. Precision=2/3, Recall=2/3"}],starterCode:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Tuple of (precision, recall) rounded to 4 decimal places
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed predictions",input:"([1, 1, 0, 1, 0], [1, 0, 0, 1, 1])",expected:"(0.6667, 0.6667)",hidden:!1},{id:"2",description:"Perfect predictions",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"(1.0, 1.0)",hidden:!1},{id:"3",description:"High precision, low recall",input:"([1, 1, 1, 1, 0], [1, 0, 0, 0, 0])",expected:"(1.0, 0.25)",hidden:!0}],hints:["Count TP where both y_true and y_pred are 1.","Count FP where y_pred is 1 but y_true is 0.","Count FN where y_true is 1 but y_pred is 0."],solution:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return (round(precision, 4), round(recall, 4))
`},{id:"cross-validation",title:"K-Fold Split Indices",section:"model-evaluation",difficulty:"medium",description:`
## K-Fold Cross-Validation Indices

Generate train and validation indices for K-fold cross-validation.

### Concept
Split data into K equal parts. For each fold:
- Use 1 part for validation
- Use remaining K-1 parts for training

### Function Signature
\`\`\`python
def kfold_indices(n_samples: int, k: int) -> list:
\`\`\`

Returns list of (train_indices, val_indices) tuples for each fold.
    `,examples:[{input:"n_samples = 6, k = 3",output:"[([2,3,4,5], [0,1]), ([0,1,4,5], [2,3]), ([0,1,2,3], [4,5])]",explanation:"Split [0,1,2,3,4,5] into 3 folds of size 2"}],starterCode:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.

    Args:
        n_samples: Total number of samples
        k: Number of folds

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic 3-fold",input:"(6, 3)",expected:"[[[2, 3, 4, 5], [0, 1]], [[0, 1, 4, 5], [2, 3]], [[0, 1, 2, 3], [4, 5]]]",hidden:!1},{id:"2",description:"2-fold",input:"(4, 2)",expected:"[[[2, 3], [0, 1]], [[0, 1], [2, 3]]]",hidden:!1},{id:"3",description:"4-fold",input:"(8, 4)",expected:"[[[2, 3, 4, 5, 6, 7], [0, 1]], [[0, 1, 4, 5, 6, 7], [2, 3]], [[0, 1, 2, 3, 6, 7], [4, 5]], [[0, 1, 2, 3, 4, 5], [6, 7]]]",hidden:!0}],hints:["Divide indices into k equal-sized chunks.","For each fold, one chunk is validation, rest are training.","Use np.array_split() to create chunks."],solution:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_indices = list(range(current, current + fold_size))
        train_indices = [i for i in range(n_samples) if i not in val_indices]
        folds.append((train_indices, val_indices))
        current += fold_size

    return folds
`},{id:"confusion-matrix",title:"Accuracy Score",section:"model-evaluation",difficulty:"easy",description:`
## Accuracy Score

Calculate the accuracy of predictions - the proportion of correct predictions.

### Formula
\`\`\`
Accuracy = (Number of correct predictions) / (Total predictions)
\`\`\`

### Function Signature
\`\`\`python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
\`\`\`

### Note
While simple, accuracy can be misleading for imbalanced datasets.
    `,examples:[{input:"y_true = [0, 1, 1, 0, 1], y_pred = [0, 1, 0, 0, 1]",output:"0.8",explanation:"4 correct out of 5 predictions"}],starterCode:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic accuracy",input:"([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])",expected:"0.8",hidden:!1},{id:"2",description:"Perfect accuracy",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"1.0",hidden:!1},{id:"3",description:"Zero accuracy",input:"([0, 0, 1, 1], [1, 1, 0, 0])",expected:"0.0",hidden:!0}],hints:["Compare y_true and y_pred element-wise.","Count matches and divide by total.","np.mean() of boolean array gives proportion of True."],solution:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
`}],rb=[{id:"mlp-forward-backward",title:"MLP Forward & Backward Pass",section:"neural-networks",difficulty:"hard",description:"\n## MLP Forward & Backward Pass\n\nImplement both forward and backward passes for a 2-layer MLP (Multi-Layer Perceptron).\n\n### Architecture\n```\nInput (n_features) → Hidden (n_hidden) → Output (n_classes)\n```\n\n### Forward Pass\n```\nZ1 = X @ W1 + b1\nA1 = ReLU(Z1)\nZ2 = A1 @ W2 + b2\noutput = softmax(Z2)\n```\n\nReturn: `(output, cache)` where cache = `(X, Z1, A1, Z2, W1, W2)`\n\n### Backward Pass (Backpropagation)\nUsing the chain rule, gradients flow backwards:\n\n```\nLoss → dZ2 → dW2, db2 → dA1 → dZ1 (ReLU) → dW1, db1\n```\n\n**Key Formulas:**\n- `dZ2 = output - Y_onehot` (softmax + cross-entropy derivative)\n- `dW2 = (1/m) * A1.T @ dZ2`\n- `db2 = (1/m) * sum(dZ2, axis=0)`\n- `dZ1 = (dZ2 @ W2.T) * (Z1 > 0)` (ReLU derivative)\n- `dW1 = (1/m) * X.T @ dZ1`\n- `db1 = (1/m) * sum(dZ1, axis=0)`\n\nReturn: `{'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...}`\n    ",examples:[{input:"X shape (2, 3), hidden_size=4, output_size=2",output:"Forward: probabilities (2, 2), rows sum to 1. Backward: gradients for all weights/biases",explanation:"Forward computes predictions; backward computes gradients for training"}],starterCode:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    """
    Forward pass for a 2-layer MLP.

    Args:
        X: Input data (batch_size, n_features)
        W1: First layer weights (n_features, n_hidden)
        b1: First layer bias (n_hidden,)
        W2: Second layer weights (n_hidden, n_classes)
        b2: Second layer bias (n_classes,)

    Returns:
        output: Class probabilities (batch_size, n_classes)
        cache: Tuple (X, Z1, A1, Z2, W1, W2) for backprop
    """
    # Your code here
    pass

def mlp_backward(Y, output, cache):
    """
    Backward pass (backpropagation) for 2-layer MLP.

    Args:
        Y: One-hot encoded labels (batch_size, n_classes)
        output: Predicted probabilities from forward pass
        cache: Cached values from forward pass (X, Z1, A1, Z2, W1, W2)

    Returns:
        grads: Dictionary with dW1, db1, dW2, db2
    """
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]  # batch size

    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward: output shape correct",input:"mlp_forward(np.array([[1, 2, 3], [4, 5, 6]]), np.ones((3, 4)), np.zeros(4), np.ones((4, 2)), np.zeros(2))[0].shape",expected:"(2, 2)",hidden:!1},{id:"2",description:"Forward: probabilities sum to 1",input:"round(float(np.sum(mlp_forward(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.zeros(2), np.array([[1, 0], [0, 1]]), np.zeros(2))[0][0])), 1)",expected:"1.0",hidden:!1},{id:"3",description:"Backward: dW2 shape matches W2",input:`(lambda: (
            X := np.array([[1.0, 2.0], [3.0, 4.0]]),
            W1 := np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            b1 := np.zeros(3),
            W2 := np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            b2 := np.zeros(2),
            result := mlp_forward(X, W1, b1, W2, b2),
            output := result[0],
            cache := result[1],
            Y := np.array([[1, 0], [0, 1]]),
            grads := mlp_backward(Y, output, cache),
            grads['dW2'].shape == W2.shape
        )[-1])()`,expected:"True",hidden:!1},{id:"4",description:"Backward: gradients are non-zero",input:`(lambda: (
            X := np.array([[1.0, 2.0], [3.0, 4.0]]),
            W1 := np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            b1 := np.zeros(3),
            W2 := np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            b2 := np.zeros(2),
            result := mlp_forward(X, W1, b1, W2, b2),
            output := result[0],
            cache := result[1],
            Y := np.array([[1, 0], [0, 1]]),
            grads := mlp_backward(Y, output, cache),
            bool(np.any(grads['dW1'] != 0) and np.any(grads['dW2'] != 0))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Backward: all gradient keys present",input:`(lambda: (
            X := np.array([[1.0, 2.0]]),
            W1 := np.array([[0.1, 0.2], [0.3, 0.4]]),
            b1 := np.zeros(2),
            W2 := np.array([[0.1], [0.2]]),
            b2 := np.zeros(1),
            result := mlp_forward(X, W1, b1, W2, b2),
            grads := mlp_backward(np.array([[1]]), result[0], result[1]),
            sorted(grads.keys())
        )[-1])()`,expected:'["dW1", "dW2", "db1", "db2"]',hidden:!0}],hints:["Forward: Compute Z1 = X @ W1 + b1, then A1 = ReLU(Z1)","Forward: Compute Z2 = A1 @ W2 + b2, then output = softmax(Z2)","Forward: Store cache = (X, Z1, A1, Z2, W1, W2) for backprop","Backward: Start from output layer: dZ2 = output - Y","Backward: dW2 = (1/m) * A1.T @ dZ2, db2 = (1/m) * sum(dZ2, axis=0)","Backward: Propagate through ReLU: dZ1 = (dZ2 @ W2.T) * (Z1 > 0)","Backward: dW1 = (1/m) * X.T @ dZ1, db1 = (1/m) * sum(dZ1, axis=0)"],solution:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    # First layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second layer
    Z2 = A1 @ W2 + b2
    output = softmax(Z2)

    # Cache for backprop
    cache = (X, Z1, A1, Z2, W1, W2)

    return output, cache

def mlp_backward(Y, output, cache):
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]

    # Output layer gradient (softmax + cross-entropy)
    dZ2 = output - Y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0)

    # Hidden layer gradient
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
`},{id:"cross-entropy-loss",title:"Cross-Entropy Loss",section:"neural-networks",difficulty:"easy",description:`
## Cross-Entropy Loss

Implement the cross-entropy loss function for multi-class classification.

### Formula
\`\`\`
L = -1/m * sum(Y * log(Y_pred))
\`\`\`

Where:
- Y is one-hot encoded true labels
- Y_pred is predicted probabilities
- m is batch size

### Numerical Stability
Add small epsilon to avoid log(0):
\`\`\`python
np.log(Y_pred + 1e-15)
\`\`\`
    `,examples:[{input:"Y_pred = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], Y = [[1,0,0], [0,1,0]]",output:"0.2231",explanation:"-1/2 * (log(0.7) + log(0.8))"}],starterCode:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    """
    Compute cross-entropy loss.

    Args:
        Y_pred: Predicted probabilities (batch_size, n_classes)
        Y_true: One-hot encoded true labels (batch_size, n_classes)

    Returns:
        loss: Scalar cross-entropy loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect prediction",input:"abs(cross_entropy_loss(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1, 0], [0, 1]])))",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"cross_entropy_loss(np.array([[0.7, 0.3], [0.2, 0.8]]), np.array([[1, 0], [0, 1]]))",expected:"0.2899",hidden:!1}],hints:["Use np.log with a small epsilon for numerical stability","Multiply element-wise: Y_true * np.log(Y_pred)","Sum and negate, then divide by batch size"],solution:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_pred.shape[0]
    epsilon = 1e-15
    log_probs = np.log(Y_pred + epsilon)
    loss = -np.sum(Y_true * log_probs) / m
    return round(loss, 4)
`},{id:"weight-init",title:"Xavier/He Weight Initialization",section:"neural-networks",difficulty:"medium",description:`
## Weight Initialization

Implement Xavier and He weight initialization to prevent vanishing/exploding gradients.

### Xavier Initialization (for tanh/sigmoid)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / (n_in + n_out))
\`\`\`

### He Initialization (for ReLU)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / n_in)
\`\`\`

### Why It Matters
- Keeps variance of activations stable across layers
- Prevents gradients from vanishing or exploding
- Enables training of deep networks
    `,examples:[{input:'n_in=784, n_out=256, method="he"',output:"Weights with std ≈ 0.0505",explanation:"sqrt(2/784) ≈ 0.0505"}],starterCode:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    """
    Initialize weights using Xavier or He initialization.

    Args:
        n_in: Number of input units
        n_out: Number of output units
        method: 'xavier' or 'he'

    Returns:
        std: Standard deviation of initialized weights (rounded to 4 decimals)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Xavier std correct",input:'(100, 50, "xavier")',expected:"0.1151",hidden:!1},{id:"2",description:"He std correct",input:'(100, 50, "he")',expected:"0.1409",hidden:!1}],hints:["Use np.random.randn to generate random numbers","Xavier: multiply by sqrt(2 / (n_in + n_out))","He: multiply by sqrt(2 / n_in)"],solution:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    np.random.seed(42)

    if method == 'xavier':
        std = np.sqrt(2.0 / (n_in + n_out))
    elif method == 'he':
        std = np.sqrt(2.0 / n_in)
    else:
        raise ValueError("Method must be 'xavier' or 'he'")

    W = np.random.randn(n_in, n_out) * std
    return round(np.std(W), 4)
`},{id:"batch-norm",title:"Batch Normalization",section:"neural-networks",difficulty:"medium",description:`
## Batch Normalization

Implement batch normalization to stabilize training.

### Forward Pass
\`\`\`
1. Compute batch mean: μ = mean(x)
2. Compute batch variance: σ² = var(x)
3. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
4. Scale and shift: out = γ * x_norm + β
\`\`\`

### Benefits
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
    `,examples:[{input:"X with mean=5, var=4, gamma=1, beta=0",output:"Normalized X with mean≈0, var≈1",explanation:"BatchNorm normalizes each feature"}],starterCode:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.

    Args:
        X: Input (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability

    Returns:
        out: Normalized output
        cache: Values needed for backward pass
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(np.allclose(np.mean(batch_norm_forward(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.ones(2), np.zeros(2))[0], axis=0), 0, atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma scales output",input:"list(np.round(batch_norm_forward(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([2.0, 3.0]), np.zeros(2))[0][1], 1))",expected:"[2.0, 3.0]",hidden:!0}],hints:["Compute mean along axis=0 (batch dimension)","Compute variance along axis=0","Normalize: (X - mean) / sqrt(var + eps)","Apply scale and shift: gamma * normalized + beta"],solution:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    # Compute batch statistics
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    # Normalize
    X_norm = (X - mu) / np.sqrt(var + eps)

    # Scale and shift
    out = gamma * X_norm + beta

    # Cache for backward pass
    cache = (X, X_norm, mu, var, gamma, eps)

    return out, cache
`},{id:"dropout",title:"Dropout",section:"neural-networks",difficulty:"easy",description:`
## Dropout Regularization

Implement dropout to prevent overfitting.

### Training Mode
\`\`\`
1. Generate mask: mask = (rand < keep_prob) / keep_prob
2. Apply mask: out = x * mask
\`\`\`

### Key Points
- Randomly "drop" neurons with probability (1 - keep_prob)
- Scale by 1/keep_prob to maintain expected value
- Disabled during inference (return input unchanged)
    `,examples:[{input:"X = [[1, 2, 3, 4]], keep_prob=0.5",output:"Some values zeroed, others scaled by 2",explanation:"Half neurons dropped, rest doubled"}],starterCode:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    """
    Apply dropout to input.

    Args:
        X: Input array
        keep_prob: Probability of keeping a neuron
        training: Whether in training mode

    Returns:
        out: Output after dropout
        mask: Dropout mask (for backward pass)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Inference returns unchanged",input:"dropout_forward(np.array([[1, 2, 3, 4]]), 0.5, False)[0].tolist()",expected:"[[1, 2, 3, 4]]",hidden:!1},{id:"2",description:"Expected value preserved",input:`(lambda: (
            np.random.seed(42),
            X := np.ones((1000, 100)),
            out := dropout_forward(X, 0.8, True)[0],
            bool(abs(np.mean(out) - 1.0) < 0.1)
        )[-1])()`,expected:"True",hidden:!0}],hints:["In training: create binary mask with np.random.rand","Scale mask by 1/keep_prob (inverted dropout)","In inference: return input unchanged"],solution:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    if not training:
        return X, None

    # Create mask and scale
    mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
    out = X * mask

    return out, mask
`}],ib=[{id:"conv2d-forward",title:"2D Convolution",section:"cnn",difficulty:"hard",description:`
## 2D Convolution Operation

Implement the forward pass of a 2D convolution (no padding, stride=1).

### Operation
For each position (i, j) in the output:
\`\`\`
out[i, j] = sum(input[i:i+kH, j:j+kW] * kernel)
\`\`\`

### Output Size
\`\`\`
out_height = input_height - kernel_height + 1
out_width = input_width - kernel_width + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d(image, kernel):
    # image: (H, W)
    # kernel: (kH, kW)
    # output: (H-kH+1, W-kW+1)
\`\`\`
    `,examples:[{input:"image 4x4, kernel 3x3",output:"output 2x2",explanation:"4-3+1 = 2 in each dimension"}],starterCode:`import numpy as np

def conv2d(image, kernel):
    """
    Apply 2D convolution to an image.

    Args:
        image: Input image (H, W)
        kernel: Convolution kernel (kH, kW)

    Returns:
        output: Convolved image (H-kH+1, W-kW+1)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Identity kernel",input:"([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])",expected:"[[5]]",hidden:!1},{id:"2",description:"Edge detection",input:"([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])",expected:"[[5, 5], [5, 5]]",hidden:!1}],hints:["Use nested loops to slide the kernel over the image","At each position, compute element-wise product and sum","Output size is (H-kH+1, W-kW+1)"],solution:`import numpy as np

def conv2d(image, kernel):
    image = np.array(image)
    kernel = np.array(kernel)

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output.astype(int).tolist()
`},{id:"max-pool",title:"Max Pooling",section:"cnn",difficulty:"medium",description:`
## Max Pooling

Implement 2x2 max pooling with stride 2.

### Operation
Divide input into non-overlapping 2x2 regions and take maximum of each.

### Output Size
\`\`\`
out_height = input_height // 2
out_width = input_width // 2
\`\`\`

### Purpose
- Reduces spatial dimensions
- Provides translation invariance
- Reduces computation
    `,examples:[{input:"[[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]",output:"[[6, 8], [14, 16]]",explanation:"Max of each 2x2 region"}],starterCode:`import numpy as np

def max_pool2d(image, pool_size=2):
    """
    Apply 2D max pooling.

    Args:
        image: Input image (H, W)
        pool_size: Size of pooling window

    Returns:
        output: Pooled image (H//pool_size, W//pool_size)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"4x4 to 2x2",input:"[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",expected:"[[6, 8], [14, 16]]",hidden:!1},{id:"2",description:"With negative values",input:"[[-1, -2, -3, -4], [-5, -6, -7, -8], [-9, -10, -11, -12], [-13, -14, -15, -16]]",expected:"[[-1, -3], [-9, -11]]",hidden:!0}],hints:["Iterate with step size = pool_size","For each 2x2 region, use np.max()","Output dimensions are input dimensions // pool_size"],solution:`import numpy as np

def max_pool2d(image, pool_size=2):
    image = np.array(image)
    H, W = image.shape

    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output.astype(int).tolist()
`},{id:"flatten-layer",title:"Flatten Layer",section:"cnn",difficulty:"easy",description:`
## Flatten Layer

Implement the flatten operation that converts a 3D feature map to a 1D vector for the fully connected layer.

### Operation
\`\`\`
(batch, height, width, channels) → (batch, height * width * channels)
\`\`\`

### Usage
- Connects convolutional layers to fully connected layers
- Preserves batch dimension
    `,examples:[{input:"shape (2, 4, 4, 3)",output:"shape (2, 48)",explanation:"4 * 4 * 3 = 48 features per sample"}],starterCode:`import numpy as np

def flatten(X):
    """
    Flatten feature maps to vectors.

    Args:
        X: Input tensor (batch, height, width, channels)

    Returns:
        output: Flattened tensor (batch, height*width*channels)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct output shape",input:"flatten(np.random.randn(2, 4, 4, 3)).shape",expected:"(2, 48)",hidden:!1},{id:"2",description:"Values preserved",input:"bool(np.allclose(flatten(np.arange(24).reshape(1, 2, 3, 4)), np.arange(24).reshape(1, 24)))",expected:"True",hidden:!0}],hints:["Get batch size as X.shape[0]","Use reshape to flatten remaining dimensions","np.reshape(X, (batch_size, -1)) uses -1 to infer size"],solution:`import numpy as np

def flatten(X):
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)
`},{id:"conv-output-size",title:"Convolution Output Size",section:"cnn",difficulty:"easy",description:`
## Calculate Convolution Output Size

Implement a function to calculate the output dimensions of a convolution layer.

### Formula
\`\`\`
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
\`\`\`

### Parameters
- **input_size**: Height or width of input
- **kernel_size**: Height or width of kernel
- **padding**: Zero-padding added to input
- **stride**: Step size of kernel
    `,examples:[{input:"input=32, kernel=3, padding=1, stride=1",output:"32",explanation:"(32 - 3 + 2*1) / 1 + 1 = 32"}],starterCode:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate output size of a convolution layer.

    Args:
        input_size: Input dimension (height or width)
        kernel_size: Kernel dimension
        padding: Zero-padding
        stride: Stride

    Returns:
        output_size: Output dimension
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same padding",input:"(32, 3, 1, 1)",expected:"32",hidden:!1},{id:"2",description:"No padding, stride 2",input:"(32, 3, 0, 2)",expected:"15",hidden:!1},{id:"3",description:"Large kernel",input:"(224, 7, 3, 2)",expected:"112",hidden:!0}],hints:["Apply the formula: (input - kernel + 2*padding) / stride + 1","Use integer division (//) for the result"],solution:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1
`},{id:"conv2d-advanced",title:"Advanced 2D Convolution",section:"cnn",difficulty:"hard",description:`
## Advanced 2D Convolution

Implement a full-featured 2D convolution with **padding**, **stride**, and **groups** support.

### Parameters
- **padding**: Zero-padding added to input borders
- **stride**: Step size when sliding the kernel
- **groups**: Split input/output channels into groups (used in depthwise separable convolutions)

### Group Convolution
When \`groups > 1\`:
- Input channels are split into \`groups\` chunks
- Output channels are split into \`groups\` chunks
- Each group's input is convolved only with its corresponding kernel group
- \`in_channels\` and \`out_channels\` must be divisible by \`groups\`

### Special Cases
- \`groups=1\`: Standard convolution
- \`groups=in_channels=out_channels\`: Depthwise convolution (MobileNet)

### Output Size Formula
\`\`\`
H_out = (H_in + 2*padding - kernel_height) // stride + 1
W_out = (W_in + 2*padding - kernel_width) // stride + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    # input: (batch, in_channels, H, W)
    # kernel: (out_channels, in_channels//groups, kH, kW)
    # output: (batch, out_channels, H_out, W_out)
\`\`\`
    `,examples:[{input:"input (1, 4, 5, 5), kernel (8, 2, 3, 3), groups=2",output:"output (1, 8, 3, 3)",explanation:"Groups=2: channels 0-1 use kernels 0-3, channels 2-3 use kernels 4-7"},{input:"input (1, 1, 5, 5), kernel (1, 1, 3, 3), padding=1, stride=2",output:"output (1, 1, 3, 3)",explanation:"(5+2*1-3)//2+1 = 3"}],starterCode:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    """
    Advanced 2D convolution with padding, stride, and groups.

    Args:
        input: Input tensor (batch, in_channels, H, W)
        kernel: Convolution kernels (out_channels, in_channels//groups, kH, kW)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)

    Returns:
        output: Convolved tensor (batch, out_channels, H_out, W_out)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic convolution with padding and stride",input:"conv2d_advanced(np.ones((1, 1, 5, 5)), np.ones((1, 1, 3, 3)), 1, 2, 1).shape",expected:"(1, 1, 3, 3)",hidden:!1},{id:"2",description:"Grouped convolution (groups=2)",input:"conv2d_advanced(np.arange(32).reshape(1, 2, 4, 4).astype(float), np.ones((4, 1, 2, 2)), 0, 1, 2).shape",expected:"(1, 4, 3, 3)",hidden:!1},{id:"3",description:"Depthwise convolution (groups=in_channels)",input:"conv2d_advanced(np.ones((2, 3, 4, 4)), np.ones((3, 1, 2, 2)), 0, 1, 3).shape",expected:"(2, 3, 3, 3)",hidden:!0},{id:"4",description:"Verify correct output values with groups",input:"bool(np.allclose(conv2d_advanced(np.ones((1, 2, 3, 3)), np.ones((2, 1, 2, 2)), 0, 1, 2), np.array([[[[4., 4.], [4., 4.]], [[4., 4.], [4., 4.]]]])) )",expected:"True",hidden:!0}],hints:["First, pad the input using np.pad() along the H and W dimensions","Calculate output dimensions: H_out = (H_padded - kH) // stride + 1","For groups: split input channels into chunks of size in_channels//groups","For groups: split kernels into chunks of size out_channels//groups","Convolve each input group with its corresponding kernel group","Concatenate results along the channel dimension"],solution:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    input = np.array(input, dtype=float)
    kernel = np.array(kernel, dtype=float)

    batch, in_channels, H, W = input.shape
    out_channels, kernel_in_channels, kH, kW = kernel.shape

    # Pad input
    if padding > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    _, _, H_padded, W_padded = input.shape

    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # Initialize output
    output = np.zeros((batch, out_channels, H_out, W_out))

    # Channels per group
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    for g in range(groups):
        # Input channels for this group
        in_start = g * in_channels_per_group
        in_end = in_start + in_channels_per_group

        # Output channels for this group
        out_start = g * out_channels_per_group
        out_end = out_start + out_channels_per_group

        # Get the input slice for this group
        input_group = input[:, in_start:in_end, :, :]

        # Get the kernels for this group
        kernel_group = kernel[out_start:out_end, :, :, :]

        # Perform convolution for this group
        for b in range(batch):
            for oc in range(out_channels_per_group):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = input_group[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, out_start + oc, i, j] = np.sum(region * kernel_group[oc])

    return output
`}],ab=[{id:"scaled-dot-product-attention",title:"Scaled Dot-Product Attention",section:"transformers",difficulty:"medium",description:`
## Scaled Dot-Product Attention

Implement the core attention mechanism from "Attention Is All You Need".

### Formula
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

Where:
- Q: Queries (seq_len, d_k)
- K: Keys (seq_len, d_k)
- V: Values (seq_len, d_v)
- d_k: Dimension of keys

### Steps
1. Compute attention scores: Q @ K.T
2. Scale by sqrt(d_k)
3. Apply softmax
4. Multiply by V
    `,examples:[{input:"Q, K, V all (4, 8) - 4 tokens, 8 dims",output:"output (4, 8)",explanation:"Each token attends to all tokens"}],starterCode:`import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)

    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape correct",input:"scaled_dot_product_attention(np.random.randn(4, 8), np.random.randn(4, 8), np.random.randn(4, 8))[0].shape",expected:"(4, 8)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"bool(np.allclose(scaled_dot_product_attention(np.ones((4, 8)), np.ones((4, 8)), np.ones((4, 8)))[1].sum(axis=-1), 1.0))",expected:"True",hidden:!1}],hints:["Compute scores = Q @ K.T","Scale by sqrt(d_k) where d_k = Q.shape[-1]","Apply softmax along the last axis","Return softmax(scores) @ V"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply softmax
    attention_weights = softmax(scores)

    # Compute output
    output = attention_weights @ V

    return output, attention_weights
`},{id:"multi-head-attention",title:"Multi-Head Attention",section:"transformers",difficulty:"hard",description:`
## Multi-Head Attention

Implement multi-head attention from the Transformer architecture.

### Concept
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
\`\`\`

### Steps
1. Project Q, K, V with different learned projections for each head
2. Apply scaled dot-product attention in parallel
3. Concatenate heads
4. Project back to original dimension
    `,examples:[{input:"X (4, 64), num_heads=8, d_model=64",output:"output (4, 64)",explanation:"8 heads with d_k=8 each, concatenated to 64"}],starterCode:`import numpy as np

def multi_head_attention(Q, K, V, num_heads):
    """
    Multi-head attention mechanism.

    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        output: Multi-head attention output (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention(np.random.randn(4, 64), np.random.randn(4, 64), np.random.randn(4, 64), 8).shape",expected:"(4, 64)",hidden:!1}],hints:["Reshape Q, K, V to (seq_len, num_heads, d_k)","Apply attention for each head","Reshape back to (seq_len, d_model)"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, num_heads):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Reshape to (seq_len, num_heads, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k)
    K = K.reshape(seq_len, num_heads, d_k)
    V = V.reshape(seq_len, num_heads, d_k)

    # Transpose to (num_heads, seq_len, d_k)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Scaled dot-product attention for each head
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attention = softmax(scores)
    heads = attention @ V

    # Transpose and reshape back
    heads = heads.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
    output = heads.reshape(seq_len, d_model)

    return output
`},{id:"positional-encoding",title:"Sinusoidal Positional Encoding",section:"transformers",difficulty:"medium",description:`
## Sinusoidal Positional Encoding

Implement the positional encoding from the original Transformer paper.

### Formulas
\`\`\`
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### Purpose
- Injects position information into embeddings
- Allows model to learn relative positions
- Fixed (not learned) encodings
    `,examples:[{input:"max_len=100, d_model=512",output:"PE matrix (100, 512)",explanation:"Position encoding for 100 positions"}],starterCode:`import numpy as np

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding matrix (max_len, d_model)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape",input:"positional_encoding(50, 64).shape",expected:"(50, 64)",hidden:!1},{id:"2",description:"First position starts with sin(0)=0",input:"bool(np.isclose(positional_encoding(10, 4)[0, 0], 0.0))",expected:"True",hidden:!0}],hints:["Create position indices: np.arange(max_len)","Create dimension indices for the divisor","Use div_term = 10000^(2i/d_model)","Apply sin to even indices, cos to odd indices"],solution:`import numpy as np

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
`},{id:"layer-norm",title:"Layer Normalization",section:"transformers",difficulty:"easy",description:`
## Layer Normalization

Implement layer normalization, used in Transformers instead of batch normalization.

### Formula
\`\`\`
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
\`\`\`

### Difference from BatchNorm
- **BatchNorm**: Normalizes across batch dimension
- **LayerNorm**: Normalizes across feature dimension
- LayerNorm works better for sequence models
    `,examples:[{input:"X (batch, seq_len, features)",output:"Normalized X, each position normalized independently",explanation:"Normalize across last dimension"}],starterCode:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization.

    Args:
        X: Input (batch, seq_len, features) or (seq_len, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant

    Returns:
        output: Normalized tensor
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(np.allclose(layer_norm(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), np.ones(4), np.zeros(4)).mean(axis=-1), 0.0, atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma and beta work",input:"bool(np.allclose(layer_norm(np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([1.0, 2.0, 1.0, 2.0]), np.zeros(4)), np.array([[-1.0, -2.0, 1.0, 2.0]])))",expected:"True",hidden:!0}],hints:["Compute mean and variance along the last axis","Use keepdims=True for proper broadcasting","Apply normalization: (x - mean) / sqrt(var + eps)"],solution:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    # Compute statistics along last dimension
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * X_norm + beta
`},{id:"causal-mask",title:"Causal Attention Mask",section:"transformers",difficulty:"easy",description:`
## Causal (Autoregressive) Mask

Create a causal mask for decoder self-attention to prevent attending to future tokens.

### Mask Structure
\`\`\`
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
\`\`\`

### Usage
- Add mask to attention scores before softmax
- -inf becomes 0 after softmax
- Ensures each position only attends to previous positions
    `,examples:[{input:"seq_len=4",output:"Lower triangular mask (4, 4)",explanation:"Position i can only attend to positions <= i"}],starterCode:`import numpy as np

def create_causal_mask(seq_len):
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask (seq_len, seq_len)
               0 for allowed positions, -inf for masked
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape and pattern",input:"create_causal_mask(4).shape",expected:"(4, 4)",hidden:!1},{id:"2",description:"Lower triangular zeros, upper triangular -inf",input:'bool(np.allclose(create_causal_mask(3), np.array([[0.0, float("-inf"), float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]])))',expected:"True",hidden:!0}],hints:["Create a matrix of ones using np.ones((seq_len, seq_len))","Use np.triu to get upper triangular matrix (excluding diagonal)","Replace 1s with -inf"],solution:`import numpy as np

def create_causal_mask(seq_len):
    # Create upper triangular matrix (above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Replace 1s with -inf
    mask = np.where(mask == 1, float('-inf'), 0.0)
    return mask
`}],ob=[{id:"vae-reparameterization",title:"VAE Reparameterization Trick",section:"generative-models",difficulty:"medium",description:`
## VAE Reparameterization Trick

Implement the reparameterization trick that allows backpropagation through stochastic sampling in VAEs.

### The Problem
We want to sample z ~ N(μ, σ²), but sampling is not differentiable.

### The Solution
\`\`\`
ε ~ N(0, 1)
z = μ + σ * ε
\`\`\`

### Why It Works
- ε is sampled independently of parameters
- z is now a deterministic function of μ, σ, and ε
- Gradients can flow through μ and σ
    `,examples:[{input:"mu = [0, 1], log_var = [0, 0]",output:"z = mu + exp(0.5 * log_var) * epsilon",explanation:"log_var=0 means σ=1"}],starterCode:`import numpy as np

def reparameterize(mu, log_var):
    """
    Sample from latent distribution using reparameterization trick.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        log_var: Log variance of latent distribution (batch, latent_dim)

    Returns:
        z: Sampled latent vectors (batch, latent_dim)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"reparameterize(np.zeros((2, 4)), np.zeros((2, 4))).shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Zero variance case has correct shape",input:"reparameterize(np.ones((3, 5)), np.full((3, 5), -1000)).shape",expected:"(3, 5)",hidden:!0}],hints:["std = exp(0.5 * log_var)","Sample epsilon from standard normal","z = mu + std * epsilon"],solution:`import numpy as np

def reparameterize(mu, log_var):
    np.random.seed(42)
    # Compute standard deviation
    std = np.exp(0.5 * log_var)
    # Sample epsilon from standard normal
    eps = np.random.randn(*mu.shape)
    # Reparameterization trick
    z = mu + std * eps
    return z
`},{id:"vae-loss",title:"VAE Loss Function",section:"generative-models",difficulty:"medium",description:`
## VAE Loss (ELBO)

Implement the VAE loss function: reconstruction loss + KL divergence.

### Loss Components
\`\`\`
L = L_reconstruction + L_KL

L_reconstruction = MSE(x, x_reconstructed)
L_KL = -0.5 * sum(1 + log_var - mu² - exp(log_var))
\`\`\`

### Intuition
- **Reconstruction loss**: Output should match input
- **KL divergence**: Latent distribution should be close to N(0,1)
    `,examples:[{input:"Perfect reconstruction, mu=0, var=1",output:"Loss ≈ 0",explanation:"Both terms are minimized"}],starterCode:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    """
    Compute VAE loss (negative ELBO).

    Args:
        x: Original input (batch, features)
        x_reconstructed: Reconstructed input (batch, features)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect case",input:"(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 2)), np.zeros((2, 2)))",expected:"(0.0, 0.0, 0.0)",hidden:!1},{id:"2",description:"Non-zero KL",input:"(np.zeros((1, 4)), np.zeros((1, 4)), np.ones((1, 2)), np.zeros((1, 2)))",expected:"(0.3161, 0.0, 0.3161)",hidden:!0}],hints:["Reconstruction: np.mean((x - x_reconstructed)²)","KL: -0.5 * sum(1 + log_var - mu² - exp(log_var))","Average over batch"],solution:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = np.mean((x - x_reconstructed) ** 2)

    # KL divergence
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

    total_loss = recon_loss + kl_loss

    return round(total_loss, 4), round(recon_loss, 4), round(kl_loss, 4)
`},{id:"diffusion-noise-schedule",title:"Diffusion Noise Schedule",section:"generative-models",difficulty:"easy",description:`
## Diffusion Noise Schedule

Implement a linear noise schedule for diffusion models.

### Linear Schedule
\`\`\`
β_t = β_start + t * (β_end - β_start) / T

where t = 0, 1, ..., T-1
\`\`\`

### Derived Quantities
\`\`\`
α_t = 1 - β_t
ᾱ_t = prod(α_1, ..., α_t)  # cumulative product
\`\`\`

These control how much noise is added at each step.
    `,examples:[{input:"T=1000, beta_start=0.0001, beta_end=0.02",output:"betas, alphas, alpha_bars arrays",explanation:"Standard DDPM schedule"}],starterCode:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for diffusion.

    Args:
        T: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values (T,)
        alphas: Alpha values (T,)
        alpha_bars: Cumulative alpha product (T,)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Beta range correct",input:"linear_noise_schedule(100, 0.0001, 0.02)[0].shape",expected:"(100,)",hidden:!1},{id:"2",description:"Alpha bar decreases - last < first",input:"bool(linear_noise_schedule(50, 0.001, 0.01)[2][-1] < linear_noise_schedule(50, 0.001, 0.01)[2][0])",expected:"True",hidden:!0}],hints:["Use np.linspace for linear interpolation","alphas = 1 - betas","alpha_bars = np.cumprod(alphas)"],solution:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    # Linear schedule
    betas = np.linspace(beta_start, beta_end, T)

    # Compute alphas
    alphas = 1 - betas

    # Cumulative product
    alpha_bars = np.cumprod(alphas)

    return betas, alphas, alpha_bars
`},{id:"diffusion-forward",title:"Diffusion Forward Process",section:"generative-models",difficulty:"medium",description:`
## Diffusion Forward Process (Adding Noise)

Implement the forward diffusion process that adds noise to data.

### Formula
\`\`\`
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

where ε ~ N(0, I)
\`\`\`

### Intuition
- As t increases, ᾱ_t decreases
- More noise is added, signal is reduced
- At t=T, x_T ≈ pure noise
    `,examples:[{input:"x_0 (image), t=500, T=1000",output:"x_t (noisy image)",explanation:"Halfway through diffusion process"}],starterCode:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    """
    Add noise to data using forward diffusion.

    Args:
        x_0: Original data (batch, ...)
        t: Timestep (int)
        alpha_bars: Cumulative alpha products (T,)

    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    np.random.seed(42)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"diffusion_forward(np.ones((2, 4)), 50, np.linspace(0.99, 0.01, 100))[0].shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Noise shape matches input",input:"diffusion_forward(np.ones((3, 5)), 10, np.linspace(0.99, 0.01, 100))[1].shape",expected:"(3, 5)",hidden:!0}],hints:["Get alpha_bar_t = alpha_bars[t]","Sample noise from standard normal","Apply formula: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"],solution:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    np.random.seed(42)

    alpha_bar_t = alpha_bars[t]

    # Sample noise
    noise = np.random.randn(*x_0.shape)

    # Forward process
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
`},{id:"vqvae-quantization",title:"VQ-VAE Vector Quantization",section:"generative-models",difficulty:"hard",description:`
## VQ-VAE Vector Quantization

Implement the vector quantization layer used in VQ-VAE (Vector Quantized Variational Autoencoder).

### Overview
Unlike standard VAEs that use continuous latent spaces, VQ-VAE uses **discrete** latent representations by mapping encoder outputs to the nearest embedding in a learned codebook.

### Quantization Process
1. **Encoder output**: z_e of shape (batch, H, W, D)
2. **Codebook**: K embedding vectors of dimension D
3. **Find nearest**: For each spatial position, find the closest codebook entry
4. **Quantize**: Replace z_e with the nearest codebook vector z_q

### Distance Calculation
\`\`\`
distances[b, h, w, k] = ||z_e[b, h, w] - codebook[k]||²
\`\`\`

### Straight-Through Estimator
During training, gradients flow through z_q to z_e by copying gradients:
\`\`\`
z_q = z_e + stop_gradient(z_q - z_e)
\`\`\`
(We don't implement this here, just the forward pass)

### VQ-VAE Loss Components
\`\`\`
L = reconstruction + β * ||sg[z_e] - e||² + ||z_e - sg[e]||²
\`\`\`
- **Codebook loss**: Moves embeddings toward encoder outputs
- **Commitment loss**: Commits encoder to embeddings

### Function Signature
\`\`\`python
def vq_quantize(z_e, codebook):
    # z_e: (batch, H, W, embedding_dim) - encoder output
    # codebook: (num_embeddings, embedding_dim) - K embedding vectors
    # Returns: z_q, indices
\`\`\`
    `,examples:[{input:"z_e (2, 4, 4, 64), codebook (512, 64)",output:"z_q (2, 4, 4, 64), indices (2, 4, 4)",explanation:"Each spatial position mapped to one of 512 codes"}],starterCode:`import numpy as np

def vq_quantize(z_e, codebook):
    """
    Vector quantization for VQ-VAE.

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        codebook: Embedding vectors (num_embeddings, embedding_dim)

    Returns:
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        indices: Codebook indices used (batch, H, W)
    """
    # Your code here
    pass


def vq_loss(z_e, z_q, beta=0.25):
    """
    Compute VQ-VAE losses (codebook + commitment).

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        beta: Commitment loss weight (default: 0.25)

    Returns:
        codebook_loss: Loss to update codebook
        commitment_loss: Loss to commit encoder to codebook
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"z_q output shape matches input",input:"vq_quantize(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))[0].shape",expected:"(2, 4, 4, 8)",hidden:!1},{id:"2",description:"Indices shape is correct",input:"vq_quantize(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))[1].shape",expected:"(2, 4, 4)",hidden:!1},{id:"3",description:"Indices are valid codebook indices (all less than K)",input:"bool(np.all(vq_quantize(np.random.randn(2, 3, 3, 4), np.random.randn(8, 4))[1] < 8))",expected:"True",hidden:!0},{id:"4",description:"VQ loss returns tuple of two values",input:"len(vq_loss(np.random.randn(2, 3, 3, 4), np.random.randn(2, 3, 3, 4), 0.25))",expected:"2",hidden:!0}],hints:["Reshape z_e to (batch*H*W, D) for easier distance computation","Use broadcasting: ||a - b||² = ||a||² + ||b||² - 2*a·b","np.argmin along the codebook axis gives indices","Index into codebook with the indices to get z_q","Reshape back to original spatial dimensions","For loss: codebook_loss = ||z_e.detach() - z_q||², commitment = ||z_e - z_q.detach()||²"],solution:`import numpy as np

def vq_quantize(z_e, codebook):
    z_e = np.array(z_e)
    codebook = np.array(codebook)

    batch, H, W, D = z_e.shape
    K, _ = codebook.shape

    # Flatten spatial dimensions: (batch*H*W, D)
    z_flat = z_e.reshape(-1, D)

    # Compute distances using: ||a-b||² = ||a||² + ||b||² - 2*a·b
    # z_flat: (N, D), codebook: (K, D)
    z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
    codebook_sq = np.sum(codebook ** 2, axis=1)        # (K,)
    cross = z_flat @ codebook.T                        # (N, K)

    distances = z_sq + codebook_sq - 2 * cross         # (N, K)

    # Find nearest codebook entry
    indices_flat = np.argmin(distances, axis=1)        # (N,)

    # Get quantized vectors
    z_q_flat = codebook[indices_flat]                  # (N, D)

    # Reshape back
    z_q = z_q_flat.reshape(batch, H, W, D)
    indices = indices_flat.reshape(batch, H, W)

    return z_q, indices


def vq_loss(z_e, z_q, beta=0.25):
    z_e = np.array(z_e)
    z_q = np.array(z_q)

    # Codebook loss: ||sg[z_e] - z_q||² (moves codebook toward encoder output)
    # In practice, sg[z_e] means z_e is treated as constant
    codebook_loss = np.mean((z_e - z_q) ** 2)

    # Commitment loss: ||z_e - sg[z_q]||² (commits encoder to codebook)
    # Same computation, but gradient only flows to z_e
    commitment_loss = beta * np.mean((z_e - z_q) ** 2)

    return round(codebook_loss, 4), round(commitment_loss, 4)
`},{id:"kl-divergence",title:"KL Divergence (Gaussians)",section:"generative-models",difficulty:"easy",description:`
## KL Divergence Between Gaussians

Compute the KL divergence between two univariate Gaussian distributions.

### Formula
\`\`\`
KL(P || Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
\`\`\`

Where P = N(μ_p, σ_p²) and Q = N(μ_q, σ_q²)

### Properties
- KL ≥ 0
- KL = 0 iff P = Q
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
    `,examples:[{input:"P = N(0, 1), Q = N(0, 1)",output:"0",explanation:"Same distribution"}],starterCode:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussians.

    Args:
        mu_p, sigma_p: Mean and std of P
        mu_q, sigma_q: Mean and std of Q

    Returns:
        kl: KL(P || Q)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same distribution",input:"(0, 1, 0, 1)",expected:"0.0",hidden:!1},{id:"2",description:"Different means",input:"(1, 1, 0, 1)",expected:"0.5",hidden:!1},{id:"3",description:"Different variances",input:"(0, 2, 0, 1)",expected:"0.8069",hidden:!0}],hints:["Apply the formula directly","Use np.log for natural logarithm","Remember: σ² is variance, σ is std"],solution:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = -0.5

    kl = term1 + term2 + term3
    return round(kl, 4)
`}],sb=[{id:"numpy-array-creation",title:"Array Creation Methods",section:"numpy-fundamentals",difficulty:"easy",description:`
## NumPy Array Creation

Implement a function that creates various NumPy arrays using different methods.

### Requirements
Create and return a dictionary with:
- \`zeros\`: 3x4 array of zeros
- \`ones\`: 2x3 array of ones
- \`arange\`: array from 0 to 9
- \`linspace\`: 5 evenly spaced values from 0 to 1
- \`eye\`: 4x4 identity matrix

### Function Signature
\`\`\`python
def create_arrays() -> dict:
\`\`\`
    `,examples:[{input:"create_arrays()",output:"{'zeros': array([[0,0,0,0],...]), ...}",explanation:"Dictionary with 5 different array types"}],starterCode:`import numpy as np

def create_arrays() -> dict:
    """
    Create various NumPy arrays.

    Returns:
        Dictionary with keys: zeros, ones, arange, linspace, eye
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape correct",input:'create_arrays()["zeros"].shape',expected:"(3, 4)",hidden:!1},{id:"2",description:"Linspace length correct",input:'len(create_arrays()["linspace"])',expected:"5",hidden:!1},{id:"3",description:"Eye is identity",input:'bool(np.allclose(create_arrays()["eye"], np.eye(4)))',expected:"True",hidden:!0}],hints:["Use np.zeros((3, 4)) for zero array","np.linspace(start, stop, num) creates evenly spaced values","np.eye(n) creates an identity matrix"],solution:`import numpy as np

def create_arrays() -> dict:
    return {
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4)
    }
`},{id:"numpy-indexing",title:"Advanced Indexing",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Advanced Indexing

Implement various indexing operations on a 2D array.

### Given
\`\`\`python
arr = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])
\`\`\`

### Requirements
Return a dictionary with:
- \`row_1\`: Second row (index 1)
- \`col_2\`: Third column (index 2)
- \`subarray\`: Rows 1-2, columns 2-4
- \`diagonal\`: Main diagonal
- \`reversed\`: Array with rows reversed

### Function Signature
\`\`\`python
def advanced_indexing(arr: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr = np.arange(20).reshape(4, 5)",output:"{'row_1': [5,6,7,8,9], ...}",explanation:"Various slicing and indexing operations"}],starterCode:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    """
    Perform advanced indexing operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with indexed results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row extraction",input:'advanced_indexing(np.arange(20).reshape(4, 5))["row_1"].tolist()',expected:"[5, 6, 7, 8, 9]",hidden:!1},{id:"2",description:"Subarray correct",input:'advanced_indexing(np.arange(20).reshape(4, 5))["subarray"].tolist()',expected:"[[7, 8, 9], [12, 13, 14]]",hidden:!1}],hints:["arr[1] or arr[1, :] gets row 1","arr[:, 2] gets column 2","arr[1:3, 2:5] gets a subarray","np.diag(arr) extracts diagonal","arr[::-1] reverses rows"],solution:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    return {
        'row_1': arr[1],
        'col_2': arr[:, 2],
        'subarray': arr[1:3, 2:5],
        'diagonal': np.diag(arr),
        'reversed': arr[::-1]
    }
`},{id:"numpy-broadcasting",title:"Broadcasting Operations",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Broadcasting

Implement operations that leverage NumPy's broadcasting rules.

### Task
Given a 2D array of shape (3, 4):
1. Subtract the row means from each row (center rows)
2. Subtract the column means from each column (center columns)
3. Add a 1D bias vector of shape (4,) to each row
4. Multiply by a column vector of shape (3, 1)

### Broadcasting Rules
- Arrays are compatible when dimensions are equal or one of them is 1
- Operations proceed element-wise on matching dimensions

### Function Signature
\`\`\`python
def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr (3,4), bias (4,), scale (3,1)",output:"{'row_centered': ..., 'col_centered': ..., 'biased': ..., 'scaled': ...}",explanation:"Various broadcasting operations"}],starterCode:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    """
    Perform broadcasting operations.

    Args:
        arr: 2D array of shape (3, 4)
        bias: 1D array of shape (4,)
        scale: 2D array of shape (3, 1)

    Returns:
        Dictionary with results of broadcasting operations
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row centering",input:'bool(np.allclose(broadcasting_ops(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]), np.zeros(4), np.ones((3,1)))["row_centered"].mean(axis=1), 0))',expected:"True",hidden:!1},{id:"2",description:"Bias addition shape",input:'broadcasting_ops(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]), np.array([1,1,1,1]), np.ones((3,1)))["biased"].shape',expected:"(3, 4)",hidden:!1}],hints:["Row means: arr.mean(axis=1, keepdims=True)","Column means: arr.mean(axis=0, keepdims=True) or arr.mean(axis=0)","Bias shape (4,) broadcasts with arr shape (3, 4)","Scale shape (3, 1) broadcasts with arr shape (3, 4)"],solution:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    # Row centered: subtract mean of each row
    row_means = arr.mean(axis=1, keepdims=True)  # (3, 1)
    row_centered = arr - row_means

    # Column centered: subtract mean of each column
    col_means = arr.mean(axis=0)  # (4,)
    col_centered = arr - col_means

    # Add bias to each row
    biased = arr + bias  # bias (4,) broadcasts to (3, 4)

    # Scale each row
    scaled = arr * scale  # scale (3, 1) broadcasts to (3, 4)

    return {
        'row_centered': row_centered,
        'col_centered': col_centered,
        'biased': biased,
        'scaled': scaled
    }
`},{id:"numpy-aggregations",title:"Aggregation Functions",section:"numpy-fundamentals",difficulty:"easy",description:"\n## NumPy Aggregations\n\nImplement common aggregation operations along different axes.\n\n### Task\nGiven a 2D array, compute:\n- Global statistics (mean, std, min, max, sum)\n- Row-wise statistics (along axis=1)\n- Column-wise statistics (along axis=0)\n- Argmax and argmin (indices of max/min values)\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- **Global**: `'global_mean'`, `'global_std'`, `'global_min'`, `'global_max'`, `'global_sum'`\n- **Row-wise**: `'row_mean'`, `'row_sum'`\n- **Column-wise**: `'col_mean'`, `'col_sum'`\n- **Indices**: `'argmax'` (2D index tuple), `'argmin'` (2D index tuple)\n    ",examples:[{input:"np.array([[1, 2, 3], [4, 5, 6]])",output:"{'global_mean': 3.5, 'global_sum': 21, 'row_sum': [6, 15], 'col_mean': [2.5, 3.5, 4.5], 'argmax': (1, 2), ...}",explanation:"Global, row-wise, and column-wise aggregations"}],starterCode:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    """
    Compute aggregation statistics.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with aggregation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Global mean",input:'compute_aggregations(np.array([[1, 2, 3], [4, 5, 6]]))["global_mean"]',expected:"3.5",hidden:!1},{id:"2",description:"Row sums",input:'compute_aggregations(np.array([[1, 2, 3], [4, 5, 6]]))["row_sum"].tolist()',expected:"[6, 15]",hidden:!1}],hints:["arr.mean() computes global mean","arr.mean(axis=0) computes column means","arr.mean(axis=1) computes row means","np.argmax(arr) gives index of max in flattened array"],solution:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    return {
        'global_mean': arr.mean(),
        'global_std': arr.std(),
        'global_min': arr.min(),
        'global_max': arr.max(),
        'global_sum': arr.sum(),
        'row_mean': arr.mean(axis=1),
        'row_sum': arr.sum(axis=1),
        'col_mean': arr.mean(axis=0),
        'col_sum': arr.sum(axis=0),
        'argmax': np.unravel_index(arr.argmax(), arr.shape),
        'argmin': np.unravel_index(arr.argmin(), arr.shape),
    }
`},{id:"numpy-reshape-transpose",title:"Reshape and Transpose",section:"numpy-fundamentals",difficulty:"medium",description:`
## Reshape and Transpose Operations

Master array shape manipulation - essential for ML data pipelines.

### Task
Given a 1D array of 24 elements:
1. Reshape to (4, 6)
2. Reshape to (2, 3, 4)
3. Transpose the 2D version
4. Swap axes on the 3D version
5. Flatten back to 1D

### Important Concepts
- \`reshape\` changes shape without changing data order
- \`transpose\` swaps axes
- \`-1\` in reshape means "infer this dimension"

### Expected Return Format
Return a dictionary with these keys:
- \`'arr_2d'\`: Reshaped to (4, 6)
- \`'arr_3d'\`: Reshaped to (2, 3, 4)
- \`'arr_2d_transposed'\`: Transpose of arr_2d, shape (6, 4)
- \`'arr_3d_swapped'\`: arr_3d with axes swapped (2,1,0), shape (4, 3, 2)
- \`'arr_flat'\`: Flattened back to 1D, shape (24,)
    `,examples:[{input:"np.arange(24)",output:"{'arr_2d': shape (4,6), 'arr_3d': shape (2,3,4), 'arr_2d_transposed': shape (6,4), ...}",explanation:"Various reshape and transpose operations"}],starterCode:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    """
    Perform reshape and transpose operations.

    Args:
        arr: 1D array of 24 elements

    Returns:
        Dictionary with reshaped arrays
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2D reshape shape",input:'reshape_transpose(np.arange(24))["arr_2d"].shape',expected:"(4, 6)",hidden:!1},{id:"2",description:"3D reshape shape",input:'reshape_transpose(np.arange(24))["arr_3d"].shape',expected:"(2, 3, 4)",hidden:!1}],hints:["arr.reshape(4, 6) or arr.reshape(4, -1)","arr.reshape(2, 3, 4) for 3D",".T or .transpose() for transpose",".flatten() or .ravel() for 1D"],solution:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    # 2D reshape
    arr_2d = arr.reshape(4, 6)

    # 3D reshape
    arr_3d = arr.reshape(2, 3, 4)

    # Transpose 2D
    arr_2d_T = arr_2d.T

    # Swap axes on 3D (swap axis 0 and 2)
    arr_3d_swapped = arr_3d.transpose(2, 1, 0)

    # Flatten
    arr_flat = arr_2d.flatten()

    return {
        'arr_2d': arr_2d,
        'arr_3d': arr_3d,
        'arr_2d_transposed': arr_2d_T,
        'arr_3d_swapped': arr_3d_swapped,
        'arr_flat': arr_flat
    }
`}],lb=[{id:"einsum-basics",title:"Einsum Fundamentals",section:"einsum",difficulty:"easy",description:`
## Einsum Fundamentals

Einstein summation notation provides a powerful way to express array operations.

### Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

### Basic Operations
- \`'i->'\`: Sum all elements (reduce)
- \`'ij->i'\`: Sum along columns (row sums)
- \`'ij->j'\`: Sum along rows (column sums)
- \`'ii->i'\`: Extract diagonal
- \`'ij->ji'\`: Transpose

### Task
Implement basic einsum operations on a 2D array.

### Expected Return Format
Return a dictionary with these keys:
- \`'sum_all'\`: Sum of all elements
- \`'row_sum'\`: Sum along each row
- \`'col_sum'\`: Sum along each column
- \`'transpose'\`: Transposed array
- \`'diagonal'\`: Diagonal elements (for square matrices, else None)
    `,examples:[{input:"np.array([[1,2],[3,4]])",output:"{'sum_all': 10, 'row_sum': [3,7], 'col_sum': [4,6], 'transpose': [[1,3],[2,4]], 'diagonal': [1,4]}",explanation:"Basic einsum operations on a 2x2 matrix"}],starterCode:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    """
    Perform basic einsum operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with einsum results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Sum all elements",input:'einsum_basics(np.array([[1, 2], [3, 4]]))["sum_all"]',expected:"10",hidden:!1},{id:"2",description:"Transpose",input:'bool(np.array_equal(einsum_basics(np.array([[1, 2], [3, 4]]))["transpose"], np.array([[1, 3], [2, 4]])))',expected:"True",hidden:!1}],hints:["'ij->' sums all elements","'ij->i' sums each row","'ij->ji' transposes","'ii->i' extracts diagonal (for square matrices)"],solution:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    return {
        'sum_all': np.einsum('ij->', arr),
        'row_sum': np.einsum('ij->i', arr),
        'col_sum': np.einsum('ij->j', arr),
        'transpose': np.einsum('ij->ji', arr),
        'diagonal': np.einsum('ii->i', arr) if arr.shape[0] == arr.shape[1] else None
    }
`},{id:"einsum-matrix-ops",title:"Matrix Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Matrix Operations with Einsum

Einsum can express matrix multiplication and more complex operations.

### Key Operations
- \`'ik,kj->ij'\`: Matrix multiplication (A @ B)
- \`'ij,ij->ij'\`: Element-wise multiplication (Hadamard)
- \`'ij,ij->'\`: Frobenius inner product (sum of element-wise product)
- \`'ij,j->i'\`: Matrix-vector multiplication
- \`'i,j->ij'\`: Outer product

### Task
Implement matrix operations using einsum.

### Expected Return Format
Return a dictionary with these keys:
- \`'matmul'\`: Matrix multiplication A @ B
- \`'matvec'\`: Matrix-vector multiplication A @ v
- \`'outer_product'\`: Outer product of v with itself
- \`'hadamard'\`: Element-wise A * A
- \`'frobenius'\`: Sum of A * A (Frobenius inner product)
- \`'trace'\`: Trace of A (if square, else None)
    `,examples:[{input:"A=[[1,2],[3,4]], B=[[5,6],[7,8]], v=[1,2]",output:"{'matmul': [[19,22],[43,50]], 'matvec': [5,11], 'outer_product': [[1,2],[2,4]], 'hadamard': [[1,4],[9,16]], 'frobenius': 30, 'trace': 5}",explanation:"Matrix operations via einsum"}],starterCode:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    """
    Perform matrix operations using einsum.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)
        v: Vector of shape (k,)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Matrix multiplication",input:'bool(np.array_equal(einsum_matrix_ops(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))["matmul"], np.array([[19,22],[43,50]])))',expected:"True",hidden:!1},{id:"2",description:"Outer product",input:'bool(np.array_equal(einsum_matrix_ops(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))["outer_product"], np.array([[1,2],[2,4]])))',expected:"True",hidden:!1}],hints:["'ik,kj->ij' contracts over k (matrix multiply)","'ij,j->i' contracts vector with columns","'i,j->ij' creates outer product (no contraction)"],solution:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    return {
        'matmul': np.einsum('ik,kj->ij', A, B),
        'matvec': np.einsum('ij,j->i', A, v),
        'outer_product': np.einsum('i,j->ij', v, v),
        'hadamard': np.einsum('ij,ij->ij', A, A),  # A * A
        'frobenius': np.einsum('ij,ij->', A, A),   # sum(A * A)
        'trace': np.einsum('ii->', A) if A.shape[0] == A.shape[1] else None
    }
`},{id:"einsum-batch-ops",title:"Batch Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Batch Operations with Einsum

Einsum shines for batch operations - multiple matrices at once.

### Batch Matrix Operations
- \`'bij,bjk->bik'\`: Batch matrix multiplication
- \`'bij,bj->bi'\`: Batch matrix-vector multiplication
- \`'bij->bji'\`: Batch transpose

### Attention-style Operations
- \`'bqd,bkd->bqk'\`: Query-Key dot products (attention scores)
- \`'bqk,bkv->bqv'\`: Weighted sum of values

### Task
Implement batch operations common in deep learning.

### Expected Return Format
Return a dictionary with these keys:
- \`'scores'\`: Raw attention scores Q @ K.T per batch, shape (batch, seq_q, seq_k)
- \`'attention_weights'\`: Softmax of scaled scores, shape (batch, seq_q, seq_k)
- \`'output'\`: Weighted sum of values, shape (batch, seq_q, dim_v)
    `,examples:[{input:"Q (2,4,8), K (2,6,8), V (2,6,16)",output:"{'scores': shape (2,4,6), 'attention_weights': shape (2,4,6), 'output': shape (2,4,16)}",explanation:"Batch attention: scores from Q@K.T, then weighted sum of V"}],starterCode:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    """
    Perform batch operations using einsum.

    Args:
        Q: Queries of shape (batch, seq_q, dim)
        K: Keys of shape (batch, seq_k, dim)
        V: Values of shape (batch, seq_k, dim_v)

    Returns:
        Dictionary with batch operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Attention scores shape",input:'einsum_batch_ops(np.ones((2, 4, 8)), np.ones((2, 6, 8)), np.ones((2, 6, 16)))["scores"].shape',expected:"(2, 4, 6)",hidden:!1},{id:"2",description:"Output shape",input:'einsum_batch_ops(np.ones((2, 4, 8)), np.ones((2, 6, 8)), np.ones((2, 6, 16)))["output"].shape',expected:"(2, 4, 16)",hidden:!1}],hints:["'bqd,bkd->bqk' computes Q @ K.T for each batch","'bqk,bkv->bqv' applies attention weights to values","b is the batch dimension, preserved in output"],solution:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    # Q: (batch, seq_q, dim)
    # K: (batch, seq_k, dim)
    # V: (batch, seq_k, dim_v)

    # Attention scores: Q @ K.T for each batch
    # Result: (batch, seq_q, seq_k)
    scores = np.einsum('bqd,bkd->bqk', Q, K)

    # Scale scores
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax (simplified, along last axis)
    exp_scores = np.exp(scaled_scores - scaled_scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # Result: (batch, seq_q, dim_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output
    }
`},{id:"einsum-advanced",title:"Advanced Einsum Patterns",section:"einsum",difficulty:"hard",description:`
## Advanced Einsum Patterns

Complex tensor operations for deep learning.

### Multi-Head Attention Pattern
\`\`\`python
# Split heads: (batch, seq, heads, dim)
# Attention per head: 'bhqd,bhkd->bhqk'
# Combine heads: 'bhqv->bqhv' then reshape
\`\`\`

### Bilinear Operations
\`\`\`python
# x.T @ W @ y: 'i,ijk,k->j'
# Batch bilinear: 'bi,ijk,bk->bj'
\`\`\`

### Task
Implement multi-head attention using einsum.
    `,examples:[{input:"Q, K, V with multiple heads",output:"Multi-head attention output",explanation:"Parallel attention across heads"}],starterCode:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    """
    Multi-head attention using einsum.

    Args:
        Q: Queries (batch, seq_q, d_model)
        K: Keys (batch, seq_k, d_model)
        V: Values (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        output: (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention_einsum(np.ones((2, 4, 64)), np.ones((2, 4, 64)), np.ones((2, 4, 64)), 8).shape",expected:"(2, 4, 64)",hidden:!1},{id:"2",description:"Output is valid array",input:"not np.isnan(multi_head_attention_einsum(np.ones((2, 4, 64)), np.ones((2, 4, 64)), np.ones((2, 4, 64)), 8)).any()",expected:"True",hidden:!0}],hints:["Reshape Q, K, V to (batch, seq, num_heads, d_k)","Use 'bqhd,bkhd->bhqk' for attention scores per head","Use 'bhqk,bkhd->bqhd' for weighted values","Reshape back to (batch, seq, d_model)"],solution:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Reshape to (batch, seq, num_heads, d_k)
    Q = Q.reshape(batch, seq_q, num_heads, d_k)
    K = K.reshape(batch, seq_k, num_heads, d_k)
    V = V.reshape(batch, seq_k, num_heads, d_k)

    # Compute attention scores for all heads
    # (batch, seq_q, heads, d_k) x (batch, seq_k, heads, d_k) -> (batch, heads, seq_q, seq_k)
    scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(d_k)

    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # (batch, heads, seq_q, seq_k) x (batch, seq_k, heads, d_k) -> (batch, seq_q, heads, d_k)
    output = np.einsum('bhqk,bkhd->bqhd', attention, V)

    # Reshape back to (batch, seq_q, d_model)
    output = output.reshape(batch, seq_q, d_model)

    return output
`},{id:"einsum-vs-matmul",title:"Einsum vs Traditional Operations",section:"einsum",difficulty:"easy",description:"\n## Einsum vs Traditional Operations\n\nCompare einsum with equivalent NumPy operations.\n\n### Equivalences\n| Einsum | NumPy |\n|--------|-------|\n| `'ij->'` | `np.sum(A)` |\n| `'ij->i'` | `np.sum(A, axis=1)` |\n| `'ik,kj->ij'` | `A @ B` |\n| `'ij,ij->'` | `np.sum(A * B)` |\n| `'ij->ji'` | `A.T` |\n\n### Task\nVerify einsum produces same results as traditional operations.\n\n### Expected Return Format\nReturn a dictionary with:\n- `'results'`: Dict mapping operation names to {'einsum': ..., 'numpy': ...}\n- `'all_match'`: Boolean indicating if all einsum results match numpy equivalents\n    ",examples:[{input:"A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]",output:"{'results': {'sum': {...}, 'matmul': {...}, ...}, 'all_match': True}",explanation:"All einsum operations produce identical results to NumPy equivalents"}],starterCode:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compare einsum with traditional NumPy operations.

    Args:
        A: First matrix (m, k)
        B: Second matrix (k, n)

    Returns:
        Dictionary with 'einsum' and 'numpy' results for comparison
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"All results match",input:'einsum_equivalence(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]), np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0]]))["all_match"]',expected:"True",hidden:!1},{id:"2",description:"Matrix multiplication matches",input:'bool(np.allclose(einsum_equivalence(np.array([[1.0,2.0],[3.0,4.0]]), np.array([[5.0,6.0],[7.0,8.0]]))["results"]["matmul"]["einsum"], np.array([[19.0,22.0],[43.0,50.0]])))',expected:"True",hidden:!1}],hints:["Use np.allclose() to compare floating point arrays","Remember 'ik,kj->ij' is matrix multiplication"],solution:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    results = {
        'sum': {
            'einsum': np.einsum('ij->', A),
            'numpy': np.sum(A)
        },
        'row_sum': {
            'einsum': np.einsum('ij->i', A),
            'numpy': np.sum(A, axis=1)
        },
        'matmul': {
            'einsum': np.einsum('ik,kj->ij', A, B),
            'numpy': A @ B
        },
        'transpose': {
            'einsum': np.einsum('ij->ji', A),
            'numpy': A.T
        },
        'hadamard_sum': {
            'einsum': np.einsum('ij,ij->', A, A),
            'numpy': np.sum(A * A)
        }
    }

    # Verify all match
    all_match = all(
        np.allclose(v['einsum'], v['numpy'])
        for v in results.values()
    )

    return {'results': results, 'all_match': all_match}
`}],ub=[{id:"tensor-creation",title:"Tensor Creation (NumPy Style)",section:"pytorch-basics",difficulty:"easy",description:"\n## Tensor Creation\n\nLearn tensor creation patterns used in PyTorch, implemented with NumPy.\n\n### PyTorch Equivalents\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.tensor([1,2,3])` | `np.array([1,2,3])` |\n| `torch.zeros(3, 4)` | `np.zeros((3, 4))` |\n| `torch.randn(2, 3)` | `np.random.randn(2, 3)` |\n| `torch.arange(10)` | `np.arange(10)` |\n| `torch.linspace(0, 1, 5)` | `np.linspace(0, 1, 5)` |\n\n### Data Types\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.float32` | `np.float32` |\n| `torch.int64` | `np.int64` |\n\n### Task\nCreate tensors matching PyTorch patterns.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- `'from_list'`: Array from [1,2,3,4] with float32 dtype\n- `'zeros'`: Zero array of shape (3, 4)\n- `'ones'`: Ones array of shape (2, 3)\n- `'randn'`: Random normal array of shape (2, 3)\n- `'arange'`: Array from 0 to 9\n- `'linspace'`: 5 evenly spaced values from 0 to 1\n- `'eye'`: 4x4 identity matrix\n- `'full'`: (2, 3) array filled with 7.0\n    ",examples:[{input:"create_tensors()",output:"{'zeros': shape (3,4), 'ones': shape (2,3), 'randn': shape (2,3), 'arange': [0..9], ...}",explanation:"Dictionary containing tensors created with various methods"}],starterCode:`import numpy as np

def create_tensors() -> dict:
    """
    Create tensors using various methods.

    Returns:
        Dictionary with named tensors
    """
    np.random.seed(42)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape",input:'create_tensors()["zeros"].shape',expected:"(3, 4)",hidden:!1},{id:"2",description:"Random tensor shape",input:'create_tensors()["randn"].shape',expected:"(2, 3)",hidden:!1}],hints:["np.zeros((3, 4)) for 3x4 zeros","np.random.randn(2, 3) for random normal","Specify dtype with astype() or dtype parameter"],solution:`import numpy as np

def create_tensors() -> dict:
    np.random.seed(42)

    return {
        'from_list': np.array([1, 2, 3, 4], dtype=np.float32),
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'randn': np.random.randn(2, 3),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4),
        'full': np.full((2, 3), 7.0),
    }
`},{id:"tensor-operations",title:"Tensor Operations",section:"pytorch-basics",difficulty:"easy",description:"\n## Basic Tensor Operations\n\nCommon tensor operations used in neural networks.\n\n### Arithmetic Operations\n- Element-wise: +, -, *, /\n- Matrix multiplication: @ or np.matmul\n- Power: ** or np.power\n\n### Reduction Operations\n- sum, mean, std, var\n- min, max, argmin, argmax\n\n### Shape Operations\n- reshape, view (reshape in NumPy)\n- squeeze, unsqueeze (np.squeeze, np.expand_dims)\n- permute, transpose\n\n### Task\nImplement common tensor operations.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- **Arithmetic**: `'add'`, `'sub'`, `'mul'`, `'div'`, `'pow'`\n- **Reductions**: `'sum_all'`, `'sum_axis0'`, `'sum_axis1'`, `'mean'`, `'std'`, `'max'`, `'argmax'`\n- **Shape ops**: `'reshape'` (to 3,2), `'flatten'`, `'unsqueeze'` (add dim 0), `'squeeze'`, `'transpose'`\n    ",examples:[{input:"x=[[1,2,3],[4,5,6]], y=[[1,1,1],[2,2,2]]",output:"{'add': [[2,3,4],[6,7,8]], 'mean': 3.5, 'reshape': [[1,2],[3,4],[5,6]], ...}",explanation:"Arithmetic, reduction, and shape operations on tensors"}],starterCode:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform common tensor operations.

    Args:
        x: First tensor (2, 3)
        y: Second tensor (2, 3)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Addition",input:'bool(np.array_equal(tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["add"], np.array([[2,3,4],[6,7,8]])))',expected:"True",hidden:!1},{id:"2",description:"Mean computation",input:'tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["mean"]',expected:"3.5",hidden:!1}],hints:["Element-wise ops work directly: x + y, x * y","Use axis parameter for reductions along specific dims","np.expand_dims(x, axis=0) adds dimension"],solution:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        # Arithmetic
        'add': x + y,
        'sub': x - y,
        'mul': x * y,
        'div': x / (y + 1e-8),
        'pow': x ** 2,

        # Reductions
        'sum_all': x.sum(),
        'sum_axis0': x.sum(axis=0),
        'sum_axis1': x.sum(axis=1),
        'mean': x.mean(),
        'std': x.std(),
        'max': x.max(),
        'argmax': x.argmax(),

        # Shape ops
        'reshape': x.reshape(3, 2),
        'flatten': x.flatten(),
        'unsqueeze': np.expand_dims(x, axis=0),  # (1, 2, 3)
        'squeeze': np.squeeze(np.expand_dims(x, 0)),
        'transpose': x.T,
    }
`},{id:"autograd-concepts",title:"Autograd Concepts (Manual)",section:"pytorch-basics",difficulty:"medium",description:`
## Autograd Concepts

Understand automatic differentiation by implementing it manually.

### PyTorch Autograd
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
\`\`\`

### Manual Gradient Computation
For y = x² + 3x + 1:
- dy/dx = 2x + 3

### Task
Implement forward pass and gradient computation for simple functions.
    `,examples:[{input:"compute_gradients(x=2.0)",output:"{'y': 11.0, 'grad': 7.0}",explanation:"Forward and backward pass"}],starterCode:`import numpy as np

def compute_gradients(x: float) -> dict:
    """
    Compute forward and backward pass for y = x^2 + 3x + 1.

    Args:
        x: Input value

    Returns:
        Dictionary with 'y' (forward result) and 'grad' (dy/dx)
    """
    # Your code here
    pass


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    """
    Compute gradients for a linear layer: Y = X @ W + b

    Args:
        X: Input (batch, in_features)
        W: Weights (in_features, out_features)
        b: Bias (out_features,)
        grad_output: Gradient from next layer (batch, out_features)

    Returns:
        Dictionary with gradients for X, W, and b
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple gradient",input:'compute_gradients(2.0)["grad"]',expected:"7.0",hidden:!1},{id:"2",description:"Forward pass result",input:'compute_gradients(2.0)["y"]',expected:"11.0",hidden:!1}],hints:["For y = x² + 3x + 1, dy/dx = 2x + 3","For Y = X @ W + b: dL/dW = X.T @ grad_output","dL/dX = grad_output @ W.T","dL/db = sum(grad_output, axis=0)"],solution:`import numpy as np

def compute_gradients(x: float) -> dict:
    # Forward pass
    y = x ** 2 + 3 * x + 1

    # Backward pass (analytical gradient)
    grad = 2 * x + 3

    return {'y': y, 'grad': grad}


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    # Forward: Y = X @ W + b

    # Backward:
    # dL/dW = X.T @ grad_output
    grad_W = X.T @ grad_output

    # dL/db = sum of grad_output over batch
    grad_b = grad_output.sum(axis=0)

    # dL/dX = grad_output @ W.T
    grad_X = grad_output @ W.T

    return {
        'grad_W': grad_W,
        'grad_b': grad_b,
        'grad_X': grad_X
    }
`},{id:"nn-modules",title:"Neural Network Modules",section:"pytorch-basics",difficulty:"medium",description:`
## Neural Network Modules

Implement PyTorch-style nn.Module patterns in NumPy.

### Module Pattern
\`\`\`python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
\`\`\`

### Task
Implement Linear and ReLU modules with forward method.
    `,examples:[{input:"Linear(10, 5).forward(x)",output:"Output of shape (batch, 5)",explanation:"Linear transformation"}],starterCode:`import numpy as np

class Linear:
    """Linear layer: Y = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        # Initialize weights with small random values
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass

    def parameters(self):
        # Return list of parameters
        pass


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


class Sequential:
    """Container for sequential layers"""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a 2-layer MLP: Linear -> ReLU -> Linear"""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Linear output shape",input:"Linear(10, 5).forward(np.random.randn(4, 10)).shape",expected:"(4, 5)",hidden:!1},{id:"2",description:"Sequential forward",input:"create_mlp(10, 20, 5).forward(np.random.randn(4, 10)).shape",expected:"(4, 5)",hidden:!1}],hints:["Initialize W with np.random.randn() * 0.01","ReLU: np.maximum(0, x)","Sequential applies layers in order"],solution:`import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim)
    )
`},{id:"loss-functions",title:"Loss Functions",section:"pytorch-basics",difficulty:"medium",description:`
## Common Loss Functions

Implement loss functions used in deep learning.

### Cross-Entropy Loss
\`\`\`
L = -sum(y * log(softmax(logits)))
\`\`\`

### MSE Loss
\`\`\`
L = mean((y_pred - y_true)²)
\`\`\`

### Binary Cross-Entropy
\`\`\`
L = -mean(y * log(p) + (1-y) * log(1-p))
\`\`\`

### Task
Implement these loss functions with numerical stability.
    `,examples:[{input:"cross_entropy(logits, targets)",output:"Scalar loss value",explanation:"Cross-entropy for classification"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # Your code here
    pass


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw scores (batch, num_classes)
        targets: Integer class labels (batch,)

    Returns:
        Scalar loss
    """
    # Your code here
    pass


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss."""
    # Your code here
    pass


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy with numerical stability."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Cross-entropy computation",input:"round(cross_entropy_loss(np.array([[2.0, 1.0, 0.1]]), np.array([0])), 4)",expected:"0.417",hidden:!1},{id:"2",description:"MSE computation",input:"mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))",expected:"0.333333",hidden:!1}],hints:["Softmax: exp(x - max(x)) / sum(exp(x - max(x)))","Use np.clip for numerical stability in log","Index logits with targets for cross-entropy"],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Clip for numerical stability
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Select probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), targets]

    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs))
    return loss


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
`}],cb=[{id:"e2e-mlp",title:"E2E: 2-Layer MLP with Backprop",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End 2-Layer MLP

Implement a complete 2-layer MLP with forward pass, backward pass, and training loop.

### Architecture
\`\`\`
Input (batch, input_dim)
  ↓
Linear(input_dim, hidden_dim) + ReLU
  ↓
Linear(hidden_dim, output_dim)
  ↓
Softmax + Cross-Entropy Loss
\`\`\`

### Forward Pass
\`\`\`python
h = relu(X @ W1 + b1)
logits = h @ W2 + b2
loss = cross_entropy(softmax(logits), y)
\`\`\`

### Backward Pass (Chain Rule)
\`\`\`python
# Output layer
d_logits = softmax(logits) - y_onehot  # (batch, output_dim)
d_W2 = h.T @ d_logits
d_b2 = d_logits.sum(axis=0)

# Hidden layer
d_h = d_logits @ W2.T
d_relu = d_h * (h > 0)  # ReLU derivative
d_W1 = X.T @ d_relu
d_b1 = d_relu.sum(axis=0)
\`\`\`

### Task
Implement forward, backward, and training step.
    `,examples:[{input:"MLP(784, 128, 10) on MNIST-like data",output:"Trained model with decreasing loss",explanation:"Full training loop"}],starterCode:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize 2-layer MLP with random weights."""
        np.random.seed(42)
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Cache for backward pass
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            X: Input (batch, input_dim)

        Returns:
            logits: Output (batch, output_dim)
        """
        # Your code here
        pass

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output (batch, output_dim)
            y: True labels (batch,) as integers

        Returns:
            loss: Scalar loss value
        """
        # Your code here
        pass

    def backward(self, y: np.ndarray) -> dict:
        """
        Backward pass.

        Args:
            y: True labels (batch,) as integers

        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        # Your code here
        pass

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single training step: forward, loss, backward, update.

        Returns:
            loss: Loss value before update
        """
        # Your code here
        pass


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    """Train MLP and return loss history."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward pass shape",input:"MLP(784, 128, 10).forward(np.random.randn(4, 784)).shape",expected:"(4, 10)",hidden:!1},{id:"2",description:"Loss decreases",input:"(lambda: (m := MLP(10, 8, 3), X := np.random.randn(16, 10), y := np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0]), losses := [m.train_step(X, y) for _ in range(50)], losses[-1] < losses[0])[-1])()",expected:"True",hidden:!1},{id:"3",description:"Gradients computed",input:'(lambda: (m := MLP(10, 8, 3), X := np.random.randn(4, 10), y := np.array([0,1,2,0]), _ := m.forward(X), _ := m.compute_loss(m.cache["logits"], y), g := m.backward(y), all(k in g for k in ["dW1", "db1", "dW2", "db2"]))[-1])()',expected:"True",hidden:!0}],hints:["Cache X, h (hidden activations), and logits in forward pass","d_logits = probs - y_onehot (softmax gradient combined with cross-entropy)","ReLU gradient: 1 if x > 0, else 0","Update: W = W - lr * dW"],solution:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Layer 1: Linear + ReLU
        z1 = X @ self.W1 + self.b1
        h = self.relu(z1)

        # Layer 2: Linear
        logits = h @ self.W2 + self.b2

        # Cache for backward
        self.cache = {'X': X, 'z1': z1, 'h': h, 'logits': logits}

        return logits

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        probs = self.softmax(logits)
        batch_size = logits.shape[0]
        correct_probs = probs[np.arange(batch_size), y]
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-15, 1)))
        self.cache['probs'] = probs
        return loss

    def backward(self, y: np.ndarray) -> dict:
        X = self.cache['X']
        h = self.cache['h']
        z1 = self.cache['z1']
        probs = self.cache['probs']
        batch_size = X.shape[0]

        # One-hot encode targets
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), y] = 1

        # Output layer gradients
        d_logits = (probs - y_onehot) / batch_size
        dW2 = h.T @ d_logits
        db2 = d_logits.sum(axis=0)

        # Hidden layer gradients
        d_h = d_logits @ self.W2.T
        d_relu = d_h * (z1 > 0)  # ReLU derivative
        dW1 = X.T @ d_relu
        db1 = d_relu.sum(axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        logits = self.forward(X)
        loss = self.compute_loss(logits, y)
        grads = self.backward(y)

        # Update weights
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

        return loss


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = MLP(input_dim, 64, output_dim)

    losses = []
    for _ in range(epochs):
        loss = model.train_step(X_train, y_train, lr)
        losses.append(loss)

    return model, losses
`},{id:"e2e-transformer",title:"E2E: Transformer Encoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Transformer Encoder

Implement a complete Transformer encoder block.

### Architecture
\`\`\`
Input Embeddings + Positional Encoding
  ↓
Multi-Head Self-Attention
  ↓
Add & LayerNorm (Residual)
  ↓
Feed-Forward Network (FFN)
  ↓
Add & LayerNorm (Residual)
  ↓
Output
\`\`\`

### Components
1. **Positional Encoding**: Sinusoidal position embeddings
2. **Multi-Head Attention**: Parallel attention heads
3. **FFN**: Two linear layers with ReLU
4. **Layer Normalization**: Normalize across features
5. **Residual Connections**: Add input to output

### Task
Implement all components and combine into encoder block.
    `,examples:[{input:"TransformerEncoder(d_model=64, nhead=8, dim_ff=256)",output:"Encoded sequence (batch, seq_len, d_model)",explanation:"Complete encoder forward pass"}],starterCode:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    # Your code here
    pass


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    # Your code here
    pass


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    # Your code here
    pass


def multi_head_attention(Q, K, V, num_heads):
    """Multi-head attention."""
    # Your code here
    pass


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear"""
    # Your code here
    pass


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Initialize transformer encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Initialize weights
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through encoder block.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Your code here
        pass


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    """Stack multiple encoder blocks."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"transformer_encoder(np.random.randn(2, 10, 64), num_layers=2, d_model=64, num_heads=8, d_ff=256).shape",expected:"(2, 10, 64)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"(lambda: (Q := np.random.randn(2, 5, 64), K := np.random.randn(2, 5, 64), V := np.random.randn(2, 5, 64), out_attn := scaled_dot_product_attention(Q, K, V), bool(np.allclose(out_attn[1].sum(axis=-1), 1.0)))[-1])()",expected:"True",hidden:!1}],hints:["Attention: softmax(QK^T / sqrt(d_k)) @ V","Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta","Residual: output = LayerNorm(x + Sublayer(x))","FFN: max(0, xW1 + b1)W2 + b2"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attention = softmax(scores, axis=-1)
    return attention @ V, attention


def multi_head_attention(Q, K, V, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    d_k = Q.shape[-1]
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_k)
    attention = softmax(scores, axis=-1)
    output = np.einsum('bhqk,bhkd->bhqd', attention, V)

    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return output


def feed_forward(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        # FFN parameters
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Multi-head self-attention + residual + layer norm
        attn_output = multi_head_attention(x, x, x, self.num_heads)
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)

        # Feed-forward + residual + layer norm
        ff_output = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        x = layer_norm(x + ff_output, self.ln2_gamma, self.ln2_beta)

        return x


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    batch, seq_len, _ = x.shape

    # Add positional encoding
    PE = positional_encoding(seq_len, d_model)
    x = x + PE

    # Stack encoder blocks
    for _ in range(num_layers):
        block = TransformerEncoderBlock(d_model, num_heads, d_ff)
        x = block.forward(x)

    return x
`},{id:"e2e-vae",title:"E2E: Variational Autoencoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Variational Autoencoder

Implement a complete VAE with encoder, decoder, reparameterization, and ELBO loss.

### Architecture
\`\`\`
Encoder: x → h → (μ, log_σ²)
         ↓
Reparameterization: z = μ + σ * ε, ε ~ N(0,1)
         ↓
Decoder: z → h → x_reconstructed
\`\`\`

### Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_hat||² + KL(q(z|x) || p(z))
KL = -0.5 * sum(1 + log_var - μ² - exp(log_var))
\`\`\`

### Task
Implement encoder, decoder, reparameterization trick, and loss computation.
    `,examples:[{input:"VAE(input_dim=784, latent_dim=20)",output:"Reconstructed images + latent samples",explanation:"Full VAE forward pass"}],starterCode:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize VAE.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        np.random.seed(42)

        # Encoder weights
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)

        # Latent space (mean and log variance)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)

        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Your code here
        pass

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        # Your code here
        pass

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstruction.
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with mu, log_var, z, x_reconstructed
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        """
        Compute VAE loss (ELBO).

        Returns:
            Dictionary with total_loss, recon_loss, kl_loss
        """
        # Your code here
        pass

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate samples by decoding random latent vectors.
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Reconstruction shape",input:'VAE(784, 256, 20).forward(np.random.randn(4, 784))["x_recon"].shape',expected:"(4, 784)",hidden:!1},{id:"2",description:"KL divergence non-negative",input:'(lambda: (vae := VAE(784, 256, 20), x := np.random.randn(4, 784), out := vae.forward(x), loss := vae.compute_loss(x, out["x_recon"], out["mu"], out["log_var"]), bool(loss["kl_loss"] >= 0))[-1])()',expected:"True",hidden:!1}],hints:["Encoder: h = relu(x @ W + b), then mu = h @ W_mu, logvar = h @ W_logvar","Reparameterize: std = exp(0.5 * log_var), z = mu + std * epsilon","KL = -0.5 * mean(1 + log_var - mu² - exp(log_var))","Reconstruction loss: MSE or BCE"],solution:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        np.random.seed(42)

        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)
        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        h = np.maximum(0, x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        x_recon = 1 / (1 + np.exp(-(h @ self.W_dec2 + self.b_dec2)))  # Sigmoid
        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return {'mu': mu, 'log_var': log_var, 'z': z, 'x_recon': x_recon}

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - x_recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

        total_loss = recon_loss + kl_loss

        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def sample(self, num_samples: int) -> np.ndarray:
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z)
`},{id:"e2e-vqvae",title:"E2E: Vector Quantized VAE",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End VQ-VAE

Implement a complete VQ-VAE (Vector Quantized Variational Autoencoder) with encoder, vector quantization, and decoder.

### Architecture
\`\`\`
Encoder: x → z_e (continuous latents)
           ↓
Vector Quantization: z_e → z_q (discrete latents from codebook)
           ↓
Decoder: z_q → x_reconstructed
\`\`\`

### Key Difference from VAE
- VAE: Continuous latent space with KL regularization
- VQ-VAE: Discrete latent space using learned codebook

### Vector Quantization
1. Encoder outputs continuous vectors z_e
2. Find nearest codebook entry for each spatial position
3. Replace z_e with quantized z_q from codebook
4. Use straight-through estimator for gradients

### Loss Function
\`\`\`
L = L_reconstruction + L_codebook + β * L_commitment

L_reconstruction = ||x - x_hat||²
L_codebook = ||sg[z_e] - e||²  (moves codebook toward encoder output)
L_commitment = ||z_e - sg[e]||²  (commits encoder to codebook)
\`\`\`
Where sg[] is stop-gradient.

### Codebook EMA Update (Alternative)
Instead of gradient updates, codebook can be updated with exponential moving average:
\`\`\`
N_i = γ * N_i + (1 - γ) * n_i     (count of assignments)
m_i = γ * m_i + (1 - γ) * sum(z_e assigned to i)
e_i = m_i / N_i
\`\`\`

### Task
Implement encoder, vector quantization layer, decoder, and all loss components.
    `,examples:[{input:"VQVAE(input_dim=784, num_embeddings=512, embedding_dim=64)",output:"Reconstructed images + discrete codes",explanation:"Full VQ-VAE forward pass"}],starterCode:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Vector Quantization layer.

        Args:
            num_embeddings: Size of codebook (K)
            embedding_dim: Dimension of each embedding
        """
        np.random.seed(42)
        # Initialize codebook with small random values
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        """
        Quantize encoder output to nearest codebook entries.

        Args:
            z_e: Encoder output (batch, height, width, embedding_dim)

        Returns:
            z_q: Quantized vectors (same shape as z_e)
            indices: Codebook indices (batch, height, width)
            distances: Distances to nearest embeddings
        """
        # Your code here
        pass

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute VQ losses.

        Args:
            z_e: Encoder output
            z_q: Quantized vectors
            beta: Commitment loss weight

        Returns:
            Dictionary with codebook_loss and commitment_loss
        """
        # Your code here
        pass


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        """
        Initialize VQ-VAE.

        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden layer dimension
            num_embeddings: Codebook size
            embedding_dim: Embedding dimension
        """
        np.random.seed(42)

        # Encoder: conv layers to get spatial feature maps
        self.enc_conv1_w = np.random.randn(hidden_dim, input_channels, 4, 4) * 0.1
        self.enc_conv1_b = np.zeros(hidden_dim)
        self.enc_conv2_w = np.random.randn(embedding_dim, hidden_dim, 4, 4) * 0.1
        self.enc_conv2_b = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder: transposed conv to reconstruct
        self.dec_conv1_w = np.random.randn(hidden_dim, embedding_dim, 4, 4) * 0.1
        self.dec_conv1_b = np.zeros(hidden_dim)
        self.dec_conv2_w = np.random.randn(input_channels, hidden_dim, 4, 4) * 0.1
        self.dec_conv2_b = np.zeros(input_channels)

        self.embedding_dim = embedding_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to continuous latent representation.

        Args:
            x: Input (batch, channels, height, width)

        Returns:
            z_e: Encoder output (batch, height', width', embedding_dim)
        """
        # Your code here (simplified: use strided conv or just reshape for demo)
        pass

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        """
        Decode quantized latents to reconstruction.

        Args:
            z_q: Quantized vectors (batch, height, width, embedding_dim)

        Returns:
            x_recon: Reconstructed input
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with z_e, z_q, indices, x_recon
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute total VQ-VAE loss.

        Returns:
            Dictionary with total_loss, recon_loss, vq_loss, commitment_loss
        """
        # Your code here
        pass

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute codebook usage statistics.

        Returns:
            usage: Count of each codebook entry used
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Quantization shape",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), z_q.shape == z_e.shape)[-1])()",expected:"True",hidden:!1},{id:"2",description:"Indices valid range",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), bool(np.all(idx >= 0) and np.all(idx < 512)))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Quantized from codebook",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), bool(np.allclose(z_q[0, 0, 0], vq.embedding[idx[0, 0, 0]])))[-1])()",expected:"True",hidden:!0},{id:"4",description:"Loss values correct",input:'(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), z_q, idx, dist := vq.quantize(z_e), losses := vq.compute_loss(z_e, z_q), bool(losses["codebook_loss"] >= 0 and losses["commitment_loss"] >= 0))[-1])()',expected:"True",hidden:!0}],hints:["For quantization: flatten spatial dims, compute distances, find argmin","Distance: ||z - e||² = ||z||² + ||e||² - 2*z·e","Straight-through: z_q = z_e + stop_grad(z_q - z_e)","Codebook loss uses z_e detached, commitment uses z_q detached","For simplified encoder/decoder, use reshape operations"],solution:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        np.random.seed(42)
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        # z_e: (batch, H, W, D)
        batch, H, W, D = z_e.shape

        # Flatten spatial dimensions
        z_flat = z_e.reshape(-1, D)  # (N, D) where N = batch * H * W

        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2*z·e
        z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
        e_sq = np.sum(self.embedding ** 2, axis=1)          # (K,)
        cross = z_flat @ self.embedding.T                    # (N, K)
        distances = z_sq + e_sq - 2 * cross                  # (N, K)

        # Find nearest embedding
        indices_flat = np.argmin(distances, axis=1)          # (N,)

        # Get quantized vectors
        z_q_flat = self.embedding[indices_flat]              # (N, D)

        # Reshape back
        z_q = z_q_flat.reshape(batch, H, W, D)
        indices = indices_flat.reshape(batch, H, W)
        min_distances = np.min(distances, axis=1).reshape(batch, H, W)

        return z_q, indices, min_distances

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Codebook loss: ||sg[z_e] - z_q||²
        # In practice, this updates the codebook to move toward encoder output
        codebook_loss = np.mean((z_e - z_q) ** 2)

        # Commitment loss: ||z_e - sg[z_q]||²
        # This commits the encoder output to stay close to codebook
        commitment_loss = beta * np.mean((z_e - z_q) ** 2)

        return {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        np.random.seed(42)

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Simplified: use linear projection instead of conv for demo
        # Encoder projects input to embedding space
        self.enc_w1 = np.random.randn(input_channels * 16, hidden_dim) * 0.1
        self.enc_b1 = np.zeros(hidden_dim)
        self.enc_w2 = np.random.randn(hidden_dim, embedding_dim) * 0.1
        self.enc_b2 = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder projects back
        self.dec_w1 = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.dec_b1 = np.zeros(hidden_dim)
        self.dec_w2 = np.random.randn(hidden_dim, input_channels * 16) * 0.1
        self.dec_b2 = np.zeros(input_channels * 16)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, channels, H, W) - assume H=W=4 for simplicity
        batch = x.shape[0]

        # Flatten spatial dimensions
        x_flat = x.reshape(batch, -1)  # (batch, channels * H * W)

        # Encode
        h = np.maximum(0, x_flat @ self.enc_w1 + self.enc_b1)  # ReLU
        z_e = x_flat @ self.enc_w1[:, :self.embedding_dim] + self.enc_b1[:self.embedding_dim]

        # Reshape to spatial format (batch, 2, 2, embedding_dim)
        z_e = z_e.reshape(batch, 1, 1, self.embedding_dim)

        return z_e

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        batch = z_q.shape[0]

        # Flatten
        z_flat = z_q.reshape(batch, -1)  # (batch, embedding_dim)

        # Decode
        h = np.maximum(0, z_flat @ self.dec_w1 + self.dec_b1)  # ReLU
        x_recon = h @ self.dec_w2 + self.dec_b2

        # Reshape to image format
        x_recon = x_recon.reshape(batch, self.input_channels, 4, 4)

        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        # Encode
        z_e = self.encode(x)

        # Quantize
        z_q, indices, distances = self.vq.quantize(z_e)

        # Straight-through estimator: gradient flows through z_q to z_e
        # z_q_st = z_e + (z_q - z_e).detach()
        # For forward pass, we just use z_q

        # Decode
        x_recon = self.decode(z_q)

        return {
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'x_recon': x_recon
        }

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Reconstruction loss
        recon_loss = np.mean((x - x_recon) ** 2)

        # VQ losses
        vq_losses = self.vq.compute_loss(z_e, z_q, beta)

        # Total loss
        total_loss = recon_loss + vq_losses['codebook_loss'] + vq_losses['commitment_loss']

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss']
        }

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        usage = np.bincount(indices.flatten(), minlength=self.vq.num_embeddings)
        return usage
`},{id:"e2e-diffusion",title:"E2E: Diffusion Model",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Diffusion Model

Implement core components of a diffusion model (DDPM-style).

### Forward Process (Adding Noise)
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### Noise Schedule
\`\`\`
β_t: Linear from β_start to β_end
α_t = 1 - β_t
α_bar_t = cumulative product of α_t
\`\`\`

### Training Objective
\`\`\`
L = ||ε - ε_θ(x_t, t)||²
\`\`\`

Model learns to predict the noise added at timestep t.

### Reverse Process (Sampling)
\`\`\`
x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-α_bar_t)) * ε_θ(x_t, t)) + σ_t * z
\`\`\`

### Task
Implement noise schedule, forward process, and reverse sampling.
    `,examples:[{input:"DiffusionModel(T=1000)",output:"Denoised samples",explanation:"Forward and reverse diffusion"}],starterCode:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize diffusion model.

        Args:
            T: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.T = T

        # Compute noise schedule
        # Your code here
        pass

    def get_noise_schedule(self) -> dict:
        """Return computed noise schedule parameters."""
        # Your code here
        pass

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        """
        Add noise to data at timestep t.

        Args:
            x_0: Original data (batch, ...)
            t: Timestep

        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        np.random.seed(42)
        # Your code here
        pass

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        Single reverse diffusion step.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            predicted_noise: Noise predicted by model

        Returns:
            x_{t-1}: Less noisy data
        """
        # Your code here
        pass

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        """Compute MSE loss between true and predicted noise."""
        # Your code here
        pass

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        """
        Generate samples using reverse diffusion.

        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise given (x_t, t)

        Returns:
            Generated samples
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Alpha bar decreases",input:'(lambda: (dm := DiffusionModel(1000), sched := dm.get_noise_schedule(), bool(sched["alpha_bars"][-1] < sched["alpha_bars"][0]))[-1])()',expected:"True",hidden:!1},{id:"2",description:"Forward noise shape",input:"(lambda: (dm := DiffusionModel(1000), x_0 := np.random.randn(4, 32), x_t, noise := dm.forward_diffusion(x_0, 500), x_t.shape == x_0.shape)[-1])()",expected:"True",hidden:!1}],hints:["betas = np.linspace(beta_start, beta_end, T)","alpha_bars = np.cumprod(1 - betas)","x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise","Reverse: use mean and add small noise (except at t=0)"],solution:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.T = T

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # For reverse process
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1 - self.alpha_bars)
        self.sqrt_alphas = np.sqrt(self.alphas)

    def get_noise_schedule(self) -> dict:
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alpha_bars': self.alpha_bars
        }

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        np.random.seed(42)
        noise = np.random.randn(*x_0.shape)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # Mean of reverse distribution
        coef1 = 1 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1 - alpha_bar)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        if t > 0:
            # Add noise (except at final step)
            sigma = np.sqrt(beta)
            noise = np.random.randn(*x_t.shape)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        return x_prev

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        return np.mean((true_noise - predicted_noise) ** 2)

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        # Start from pure noise
        x = np.random.randn(*shape)

        # Reverse diffusion
        for t in reversed(range(self.T)):
            predicted_noise = noise_predictor(x, t)
            x = self.reverse_step(x, t, predicted_noise)

        return x
`},{id:"e2e-cnn",title:"E2E: Convolutional Neural Network",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End CNN

Implement a complete CNN for image classification.

### Architecture
\`\`\`
Input (batch, channels, height, width)
  ↓
Conv2D(in_channels, 32, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Conv2D(32, 64, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Flatten
  ↓
Linear(flattened_size, 128) + ReLU
  ↓
Linear(128, num_classes)
  ↓
Softmax
\`\`\`

### Components
- 2D Convolution (no padding for simplicity)
- Max Pooling
- Flatten
- Fully Connected layers

### Task
Implement conv2d, maxpool, and combine into CNN.
    `,examples:[{input:"CNN on 28x28 grayscale images",output:"Class probabilities",explanation:"Full CNN forward pass"}],starterCode:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    2D convolution (no padding, stride=1).

    Args:
        x: Input (batch, in_channels, height, width)
        kernel: Weights (out_channels, in_channels, kH, kW)
        bias: Bias (out_channels,)

    Returns:
        output: (batch, out_channels, out_height, out_width)
    """
    # Your code here
    pass


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    2D max pooling.

    Args:
        x: Input (batch, channels, height, width)
        pool_size: Size of pooling window

    Returns:
        output: Pooled tensor
    """
    # Your code here
    pass


def flatten(x: np.ndarray) -> np.ndarray:
    """Flatten all dimensions except batch."""
    # Your code here
    pass


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Initialize CNN.

        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layer 1: in_channels -> 32
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)

        # Conv layer 2: 32 -> 64
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate flattened size after convolutions and pooling
        # Your code to compute this
        pass

        # Fully connected layers
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through CNN.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            logits: (batch, num_classes)
        """
        # Your code here
        pass


def test_cnn():
    """Test CNN with sample data."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Conv output shape",input:"conv2d(np.random.randn(2, 1, 8, 8), np.random.randn(32, 1, 3, 3), np.zeros(32)).shape",expected:"(2, 32, 6, 6)",hidden:!1},{id:"2",description:"Pool output shape",input:"maxpool2d(np.random.randn(2, 32, 8, 8), pool_size=2).shape",expected:"(2, 32, 4, 4)",hidden:!1}],hints:["Conv output size: (input - kernel + 1) for stride=1, no padding","Pool output size: input // pool_size","Use nested loops for convolution (or np.lib.stride_tricks)","Flatten: x.reshape(batch_size, -1)"],solution:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, in_ch, H, W = x.shape
    out_ch, _, kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((batch, out_ch, out_H, out_W))

    for b in range(batch):
        for oc in range(out_ch):
            for i in range(out_H):
                for j in range(out_W):
                    receptive_field = x[b, :, i:i+kH, j:j+kW]
                    output[b, oc, i, j] = np.sum(receptive_field * kernel[oc]) + bias[oc]

    return output


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    batch, channels, H, W = x.shape
    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((batch, channels, out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * pool_size
            w_start = j * pool_size
            output[:, :, i, j] = np.max(
                x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size],
                axis=(2, 3)
            )

    return output


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layers
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate size after convolutions and pooling
        # After conv1 (3x3): H-2, W-2
        # After pool1 (2x2): (H-2)//2, (W-2)//2
        # After conv2 (3x3): (H-2)//2 - 2, (W-2)//2 - 2
        # After pool2 (2x2): ((H-2)//2 - 2)//2, ((W-2)//2 - 2)//2
        h1 = (H - 2) // 2
        w1 = (W - 2) // 2
        h2 = (h1 - 2) // 2
        w2 = (w1 - 2) // 2
        flat_size = 64 * h2 * w2

        # FC layers
        self.fc1_w = np.random.randn(flat_size, 128) * 0.1
        self.fc1_b = np.zeros(128)
        self.fc2_w = np.random.randn(128, num_classes) * 0.1
        self.fc2_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Conv block 1
        x = conv2d(x, self.conv1_w, self.conv1_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Conv block 2
        x = conv2d(x, self.conv2_w, self.conv2_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Flatten
        x = flatten(x)

        # FC layers
        x = np.maximum(0, x @ self.fc1_w + self.fc1_b)  # ReLU
        x = x @ self.fc2_w + self.fc2_b  # Logits

        return x


def test_cnn():
    # Test with MNIST-like data
    cnn = CNN(input_shape=(1, 28, 28), num_classes=10)
    x = np.random.randn(4, 1, 28, 28)
    logits = cnn.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    return logits.shape == (4, 10)
`}],qh=[...Gk,...Zk,...Jk,...eb,...tb,...nb,...rb,...ib,...ab,...ob,...sb,...lb,...ub,...cb];function db(e){return qh.find(t=>t.id===e)}function pb(e){return qh.filter(t=>t.section===e)}function fb(){const{sectionId:e}=Gm(),{getProblemProgress:t,getSectionProgress:n}=$i(),r=Nr.find(o=>o.id===e),i=e?pb(e):[];if(!r)return x.jsxs("div",{className:"text-center py-12",children:[x.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Section Not Found"}),x.jsx(qt,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const a=n(r.id,r.problems.length);return x.jsxs("div",{className:"max-w-4xl mx-auto",children:[x.jsxs("div",{className:"mb-8",children:[x.jsxs("div",{className:"flex items-center gap-4 mb-4",children:[x.jsx("div",{className:"w-14 h-14 bg-white border border-gray-200 rounded-xl flex items-center justify-center text-3xl shadow-sm",children:r.icon}),x.jsxs("div",{children:[x.jsx("h1",{className:"text-2xl font-bold text-gray-900",children:r.title}),x.jsx("p",{className:"text-gray-500",children:r.description})]})]}),x.jsxs("div",{className:"flex items-center gap-4",children:[x.jsx("div",{className:"flex-1 h-2 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${a}%`}})}),x.jsxs("span",{className:"text-primary-600 font-medium",children:[a,"% complete"]})]})]}),x.jsxs("div",{className:"bg-white rounded-xl p-6 mb-8 border border-gray-200 shadow-sm",children:[x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Introduction"}),x.jsx("div",{className:"prose prose-sm max-w-none",children:x.jsx(Oh,{components:{h1:({children:o})=>x.jsx("h1",{className:"text-xl font-bold text-gray-900 mt-0 mb-4",children:o}),h2:({children:o})=>x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:o}),h3:({children:o})=>x.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:o}),p:({children:o})=>x.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:o}),code:({className:o,children:s})=>(o==null?void 0:o.includes("language-"))?x.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:x.jsx("code",{className:"text-sm text-gray-800",children:s})}):x.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm",children:s}),ul:({children:o})=>x.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:o}),li:({children:o})=>x.jsx("li",{className:"text-gray-600",children:o}),strong:({children:o})=>x.jsx("strong",{className:"text-gray-900 font-semibold",children:o})},children:r.introduction})})]}),x.jsxs("div",{children:[x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Problems"}),x.jsx("div",{className:"space-y-3",children:i.map((o,s)=>{const l=t(r.id,o.id);return x.jsxs(qt,{to:`/problem/${r.id}/${o.id}`,className:"flex items-center gap-4 bg-white rounded-lg p-4 border border-gray-200 hover:border-primary-400 hover:shadow-sm transition-all group",children:[x.jsx("div",{className:`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${l.status==="completed"?"bg-green-100 text-green-600":l.status==="in_progress"?"bg-yellow-100 text-yellow-600":"bg-gray-100 text-gray-500"}`,children:l.status==="completed"?"✓":s+1}),x.jsxs("div",{className:"flex-1",children:[x.jsx("h3",{className:"text-gray-900 font-medium group-hover:text-primary-600 transition-colors",children:o.title}),x.jsxs("div",{className:"flex items-center gap-3 mt-1",children:[x.jsx("span",{className:`text-xs px-2 py-0.5 rounded ${o.difficulty==="easy"?"bg-green-100 text-green-700":o.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:o.difficulty}),x.jsxs("span",{className:"text-gray-400 text-xs",children:[o.testCases.length," test cases"]})]})]}),x.jsx("div",{className:"text-gray-400 group-hover:text-primary-600 transition-colors",children:"→"})]},o.id)})})]})]})}var Wh={exports:{}},mb="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED",hb=mb,gb=hb;function Bh(){}function Uh(){}Uh.resetWarningCache=Bh;var yb=function(){function e(r,i,a,o,s,l){if(l!==gb){var u=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types");throw u.name="Invariant Violation",u}}e.isRequired=e;function t(){return e}var n={array:e,bigint:e,bool:e,func:e,number:e,object:e,string:e,symbol:e,any:e,arrayOf:t,element:e,elementType:e,instanceOf:t,node:e,objectOf:t,oneOf:t,oneOfType:t,shape:t,exact:t,checkPropTypes:Uh,resetWarningCache:Bh};return n.PropTypes=n,n};Wh.exports=yb();var xb=Wh.exports;const oe=ho(xb);var dt=typeof window<"u"?window:null,nc=dt===null,Ii=nc?void 0:dt.document,wt="addEventListener",kt="removeEventListener",Es="getBoundingClientRect",Kr="_a",bt="_b",Vt="_c",ya="horizontal",St=function(){return!1},_b=nc?"calc":["","-webkit-","-moz-","-o-"].filter(function(e){var t=Ii.createElement("div");return t.style.cssText="width:"+e+"calc(9px)",!!t.style.length}).shift()+"calc",Vh=function(e){return typeof e=="string"||e instanceof String},fp=function(e){if(Vh(e)){var t=Ii.querySelector(e);if(!t)throw new Error("Selector "+e+" did not match a DOM element");return t}return e},De=function(e,t,n){var r=e[t];return r!==void 0?r:n},xa=function(e,t,n,r){if(t){if(r==="end")return 0;if(r==="center")return e/2}else if(n){if(r==="start")return 0;if(r==="center")return e/2}return e},vb=function(e,t){var n=Ii.createElement("div");return n.className="gutter gutter-"+t,n},wb=function(e,t,n){var r={};return Vh(t)?r[e]=t:r[e]=_b+"("+t+"% - "+n+"px)",r},kb=function(e,t){var n;return n={},n[e]=t+"px",n},mp=function(e,t){if(t===void 0&&(t={}),nc)return{};var n=e,r,i,a,o,s,l;Array.from&&(n=Array.from(n));var u=fp(n[0]),d=u.parentNode,c=getComputedStyle?getComputedStyle(d):null,p=c?c.flexDirection:null,f=De(t,"sizes")||n.map(function(){return 100/n.length}),y=De(t,"minSize",100),_=Array.isArray(y)?y:n.map(function(){return y}),S=De(t,"maxSize",1/0),m=Array.isArray(S)?S:n.map(function(){return S}),h=De(t,"expandToMin",!1),g=De(t,"gutterSize",10),C=De(t,"gutterAlign","center"),E=De(t,"snapOffset",30),b=Array.isArray(E)?E:n.map(function(){return E}),P=De(t,"dragInterval",1),T=De(t,"direction",ya),D=De(t,"cursor",T===ya?"col-resize":"row-resize"),k=De(t,"gutter",vb),I=De(t,"elementStyle",wb),W=De(t,"gutterStyle",kb);T===ya?(r="width",i="clientX",a="left",o="right",s="clientWidth"):T==="vertical"&&(r="height",i="clientY",a="top",o="bottom",s="clientHeight");function K(q,L,M,H){var ze=I(r,L,M,H);Object.keys(ze).forEach(function(_e){q.style[_e]=ze[_e]})}function te(q,L,M){var H=W(r,L,M);Object.keys(H).forEach(function(ze){q.style[ze]=H[ze]})}function F(){return l.map(function(q){return q.size})}function X(q){return"touches"in q?q.touches[0][i]:q[i]}function re(q){var L=l[this.a],M=l[this.b],H=L.size+M.size;L.size=q/this.size*H,M.size=H-q/this.size*H,K(L.element,L.size,this[bt],L.i),K(M.element,M.size,this[Vt],M.i)}function j(q){var L,M=l[this.a],H=l[this.b];this.dragging&&(L=X(q)-this.start+(this[bt]-this.dragOffset),P>1&&(L=Math.round(L/P)*P),L<=M.minSize+M.snapOffset+this[bt]?L=M.minSize+this[bt]:L>=this.size-(H.minSize+H.snapOffset+this[Vt])&&(L=this.size-(H.minSize+this[Vt])),L>=M.maxSize-M.snapOffset+this[bt]?L=M.maxSize+this[bt]:L<=this.size-(H.maxSize-H.snapOffset+this[Vt])&&(L=this.size-(H.maxSize+this[Vt])),re.call(this,L),De(t,"onDrag",St)(F()))}function V(){var q=l[this.a].element,L=l[this.b].element,M=q[Es](),H=L[Es]();this.size=M[r]+H[r]+this[bt]+this[Vt],this.start=M[a],this.end=M[o]}function v(q){if(!getComputedStyle)return null;var L=getComputedStyle(q);if(!L)return null;var M=q[s];return M===0?null:(T===ya?M-=parseFloat(L.paddingLeft)+parseFloat(L.paddingRight):M-=parseFloat(L.paddingTop)+parseFloat(L.paddingBottom),M)}function B(q){var L=v(d);if(L===null||_.reduce(function(_e,We){return _e+We},0)>L)return q;var M=0,H=[],ze=q.map(function(_e,We){var At=L*_e/100,Tn=xa(g,We===0,We===q.length-1,C),Ln=_[We]+Tn;return At<Ln?(M+=Ln-At,H.push(0),Ln):(H.push(At-Ln),At)});return M===0?q:ze.map(function(_e,We){var At=_e;if(M>0&&H[We]-M>0){var Tn=Math.min(M,H[We]-M);M-=Tn,At=_e-Tn}return At/L*100})}function $(){var q=this,L=l[q.a].element,M=l[q.b].element;q.dragging&&De(t,"onDragEnd",St)(F()),q.dragging=!1,dt[kt]("mouseup",q.stop),dt[kt]("touchend",q.stop),dt[kt]("touchcancel",q.stop),dt[kt]("mousemove",q.move),dt[kt]("touchmove",q.move),q.stop=null,q.move=null,L[kt]("selectstart",St),L[kt]("dragstart",St),M[kt]("selectstart",St),M[kt]("dragstart",St),L.style.userSelect="",L.style.webkitUserSelect="",L.style.MozUserSelect="",L.style.pointerEvents="",M.style.userSelect="",M.style.webkitUserSelect="",M.style.MozUserSelect="",M.style.pointerEvents="",q.gutter.style.cursor="",q.parent.style.cursor="",Ii.body.style.cursor=""}function w(q){if(!("button"in q&&q.button!==0)){var L=this,M=l[L.a].element,H=l[L.b].element;L.dragging||De(t,"onDragStart",St)(F()),q.preventDefault(),L.dragging=!0,L.move=j.bind(L),L.stop=$.bind(L),dt[wt]("mouseup",L.stop),dt[wt]("touchend",L.stop),dt[wt]("touchcancel",L.stop),dt[wt]("mousemove",L.move),dt[wt]("touchmove",L.move),M[wt]("selectstart",St),M[wt]("dragstart",St),H[wt]("selectstart",St),H[wt]("dragstart",St),M.style.userSelect="none",M.style.webkitUserSelect="none",M.style.MozUserSelect="none",M.style.pointerEvents="none",H.style.userSelect="none",H.style.webkitUserSelect="none",H.style.MozUserSelect="none",H.style.pointerEvents="none",L.gutter.style.cursor=D,L.parent.style.cursor=D,Ii.body.style.cursor=D,V.call(L),L.dragOffset=X(q)-L.end}}f=B(f);var ie=[];l=n.map(function(q,L){var M={element:fp(q),size:f[L],minSize:_[L],maxSize:m[L],snapOffset:b[L],i:L},H;if(L>0&&(H={a:L-1,b:L,dragging:!1,direction:T,parent:d},H[bt]=xa(g,L-1===0,!1,C),H[Vt]=xa(g,!1,L===n.length-1,C),p==="row-reverse"||p==="column-reverse")){var ze=H.a;H.a=H.b,H.b=ze}if(L>0){var _e=k(L,T,M.element);te(_e,g,L),H[Kr]=w.bind(H),_e[wt]("mousedown",H[Kr]),_e[wt]("touchstart",H[Kr]),d.insertBefore(_e,M.element),H.gutter=_e}return K(M.element,M.size,xa(g,L===0,L===n.length-1,C),L),L>0&&ie.push(H),M});function ke(q){var L=q.i===ie.length,M=L?ie[q.i-1]:ie[q.i];V.call(M);var H=L?M.size-q.minSize-M[Vt]:q.minSize+M[bt];re.call(M,H)}l.forEach(function(q){var L=q.element[Es]()[r];L<q.minSize&&(h?ke(q):q.minSize=L)});function le(q){var L=B(q);L.forEach(function(M,H){if(H>0){var ze=ie[H-1],_e=l[ze.a],We=l[ze.b];_e.size=L[H-1],We.size=M,K(_e.element,_e.size,ze[bt],_e.i),K(We.element,We.size,ze[Vt],We.i)}})}function ct(q,L){ie.forEach(function(M){if(L!==!0?M.parent.removeChild(M.gutter):(M.gutter[kt]("mousedown",M[Kr]),M.gutter[kt]("touchstart",M[Kr])),q!==!0){var H=I(r,M.a.size,M[bt]);Object.keys(H).forEach(function(ze){l[M.a].element.style[ze]="",l[M.b].element.style[ze]=""})}})}return{setSizes:le,getSizes:F,collapse:function(L){ke(l[L])},destroy:ct,parent:d,pairs:ie}};function Ps(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)===-1&&(n[r]=e[r]);return n}var Di=function(e){function t(){e.apply(this,arguments)}return e&&(t.__proto__=e),t.prototype=Object.create(e&&e.prototype),t.prototype.constructor=t,t.prototype.componentDidMount=function(){var r=this.props;r.children;var i=r.gutter,a=Ps(r,["children","gutter"]),o=a;o.gutter=function(s,l){var u;return i?u=i(s,l):(u=document.createElement("div"),u.className="gutter gutter-"+l),u.__isSplitGutter=!0,u},this.split=mp(this.parent.children,o)},t.prototype.componentDidUpdate=function(r){var i=this,a=this.props;a.children;var o=a.minSize,s=a.sizes,l=a.collapsed,u=Ps(a,["children","minSize","sizes","collapsed"]),d=u,c=r.minSize,p=r.sizes,f=r.collapsed,y=["maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor"],_=y.map(function(h){return i.props[h]!==r[h]}).reduce(function(h,g){return h||g},!1);if(Array.isArray(o)&&Array.isArray(c)){var S=!1;o.forEach(function(h,g){S=S||h!==c[g]}),_=_||S}else Array.isArray(o)||Array.isArray(c)?_=!0:_=_||o!==c;if(_)d.minSize=o,d.sizes=s||this.split.getSizes(),this.split.destroy(!0,!0),d.gutter=function(h,g,C){return C.previousSibling},this.split=mp(Array.from(this.parent.children).filter(function(h){return!h.__isSplitGutter}),d);else if(s){var m=!1;s.forEach(function(h,g){m=m||h!==p[g]}),m&&this.split.setSizes(this.props.sizes)}Number.isInteger(l)&&(l!==f||_)&&this.split.collapse(l)},t.prototype.componentWillUnmount=function(){this.split.destroy(),delete this.split},t.prototype.render=function(){var r=this,i=this.props;i.sizes,i.minSize,i.maxSize,i.expandToMin,i.gutterSize,i.gutterAlign,i.snapOffset,i.dragInterval,i.direction,i.cursor,i.gutter,i.elementStyle,i.gutterStyle,i.onDrag,i.onDragStart,i.onDragEnd,i.collapsed;var a=i.children,o=Ps(i,["sizes","minSize","maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor","gutter","elementStyle","gutterStyle","onDrag","onDragStart","onDragEnd","collapsed","children"]),s=o;return Ft.createElement("div",Object.assign({},{ref:function(l){r.parent=l}},s),a)},t}(Ft.Component);Di.propTypes={sizes:oe.arrayOf(oe.number),minSize:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),maxSize:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),expandToMin:oe.bool,gutterSize:oe.number,gutterAlign:oe.string,snapOffset:oe.oneOfType([oe.number,oe.arrayOf(oe.number)]),dragInterval:oe.number,direction:oe.string,cursor:oe.string,gutter:oe.func,elementStyle:oe.func,gutterStyle:oe.func,onDrag:oe.func,onDragStart:oe.func,onDragEnd:oe.func,collapsed:oe.number,children:oe.arrayOf(oe.element)};Di.defaultProps={sizes:void 0,minSize:void 0,maxSize:void 0,expandToMin:void 0,gutterSize:void 0,gutterAlign:void 0,snapOffset:void 0,dragInterval:void 0,direction:void 0,cursor:void 0,gutter:void 0,elementStyle:void 0,gutterStyle:void 0,onDrag:void 0,onDragStart:void 0,onDragEnd:void 0,collapsed:void 0,children:void 0};function Hh(){const[e,t]=z.useState(!1),[n,r]=z.useState(!1),[i,a]=z.useState(null),[o,s]=z.useState([]),l=z.useRef(null),u=z.useRef(!1),d=z.useCallback(_=>{s(S=>[...S,_])},[]),c=z.useCallback(()=>{s([])},[]),p=z.useCallback(async()=>{if(!(l.current||u.current)){u.current=!0,t(!0),a(null);try{window.loadPyodide||await new Promise((S,m)=>{const h=document.createElement("script");h.src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js",h.async=!0,h.onload=()=>S(),h.onerror=()=>m(new Error("Failed to load Pyodide")),document.head.appendChild(h)});const _=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"});_.setStdout({batched:S=>{S.split(`
`).forEach(h=>{h.trim()&&d(h)})}}),_.setStderr({batched:S=>{S.split(`
`).forEach(h=>{h.trim()&&d(`[Error] ${h}`)})}}),await _.loadPackage(["numpy"]),l.current=_,r(!0),d("Python environment ready with NumPy loaded.")}catch(_){const S=_ instanceof Error?_.message:"Failed to initialize Python";a(S),d(`[Error] ${S}`)}finally{t(!1),u.current=!1}}},[d]);z.useEffect(()=>{p()},[p]);const f=z.useCallback(async _=>{if(!l.current)return{success:!1,error:"Python environment not ready"};try{return c(),await l.current.runPythonAsync(_),{success:!0}}catch(S){const m=S instanceof Error?S.message:"Execution error";return d(`[Error] ${m}`),{success:!1,error:m}}},[d,c]),y=z.useCallback(async(_,S,m)=>{if(!l.current)return S.map(g=>({id:g.id,passed:!1,description:g.description,expected:g.expected,actual:"Python environment not ready",hidden:g.hidden}));const h=[];c();try{await l.current.runPythonAsync(_);for(const g of S)try{const C=g.input.includes(m)||g.input.includes(".")||/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(g.input.trim()),E=g.input.trim().startsWith("(lambda");let b;C&&!g.input.startsWith("[")&&(!g.input.startsWith("(")||E)?b=`
import numpy as np
import json

# Evaluate expression directly
result = ${g.input}

# Convert result to string for comparison
if isinstance(result, bool):
    result_str = str(result)
elif isinstance(result, (np.bool_, )):
    result_str = str(bool(result))
elif isinstance(result, np.ndarray):
    result_str = str(result.tolist())
elif isinstance(result, tuple):
    # Convert tuple elements
    converted = []
    for item in result:
        if isinstance(item, bool):
            converted.append(item)
        elif isinstance(item, (np.bool_, )):
            converted.append(bool(item))
        elif isinstance(item, np.ndarray):
            converted.append(item.tolist())
        elif isinstance(item, (np.floating, float)):
            converted.append(round(float(item), 6))
        elif isinstance(item, (np.integer, int)):
            converted.append(int(item))
        else:
            converted.append(item)
    result_str = str(tuple(converted))
elif isinstance(result, (list, dict)):
    result_str = json.dumps(result)
elif isinstance(result, (np.floating, float)):
    result_str = str(round(float(result), 6))
elif isinstance(result, (np.integer, int)):
    result_str = str(int(result))
else:
    result_str = str(result)
result_str
`:b=`
import numpy as np
import json

# Parse input
test_input = ${g.input}

# Call the function
if isinstance(test_input, list):
    result = ${m}(np.array(test_input))
elif isinstance(test_input, tuple):
    result = ${m}(*[np.array(x) if isinstance(x, list) else x for x in test_input])
else:
    result = ${m}(test_input)

# Convert result to string for comparison
if isinstance(result, bool):
    result_str = str(result)
elif isinstance(result, (np.bool_, )):
    result_str = str(bool(result))
elif isinstance(result, np.ndarray):
    result_str = str(result.tolist())
elif isinstance(result, tuple):
    # Convert tuple elements
    converted = []
    for item in result:
        if isinstance(item, bool):
            converted.append(item)
        elif isinstance(item, (np.bool_, )):
            converted.append(bool(item))
        elif isinstance(item, np.ndarray):
            converted.append(item.tolist())
        elif isinstance(item, (np.floating, float)):
            converted.append(round(float(item), 6))
        elif isinstance(item, (np.integer, int)):
            converted.append(int(item))
        else:
            converted.append(item)
    result_str = str(tuple(converted))
elif isinstance(result, (list, dict)):
    result_str = json.dumps(result)
elif isinstance(result, (np.floating, float)):
    result_str = str(round(float(result), 6))
elif isinstance(result, (np.integer, int)):
    result_str = str(int(result))
else:
    result_str = str(result)
result_str
`;const P=await l.current.runPythonAsync(b),T=String(P),D=g.expected.replace(/\s/g,""),k=T.replace(/\s/g,""),I=D===k;h.push({id:g.id,passed:I,description:g.description,expected:g.expected,actual:T,hidden:g.hidden}),d(I?`Test ${g.id}: PASSED`:`Test ${g.id}: FAILED - Expected ${g.expected}, got ${T}`)}catch(C){const E=C instanceof Error?C.message:"Test execution error";h.push({id:g.id,passed:!1,description:g.description,expected:g.expected,actual:E,hidden:g.hidden}),d(`Test ${g.id}: ERROR - ${E}`)}}catch(g){const C=g instanceof Error?g.message:"Code execution error";return d(`[Error] ${C}`),S.map(E=>({id:E.id,passed:!1,description:E.description,expected:E.expected,actual:C,hidden:E.hidden}))}return h},[d,c]);return{isLoading:e,isReady:n,error:i,output:o,runCode:f,runTests:y,clearOutput:c}}function hp(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function bb(e){if(Array.isArray(e))return e}function Sb(e,t,n){return(t=Lb(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Cb(e,t){var n=e==null?null:typeof Symbol<"u"&&e[Symbol.iterator]||e["@@iterator"];if(n!=null){var r,i,a,o,s=[],l=!0,u=!1;try{if(a=(n=n.call(e)).next,t!==0)for(;!(l=(r=a.call(n)).done)&&(s.push(r.value),s.length!==t);l=!0);}catch(d){u=!0,i=d}finally{try{if(!l&&n.return!=null&&(o=n.return(),Object(o)!==o))return}finally{if(u)throw i}}return s}}function zb(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function gp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function yp(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?gp(Object(n),!0).forEach(function(r){Sb(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):gp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function Eb(e,t){if(e==null)return{};var n,r,i=Pb(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)===-1&&{}.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}function Pb(e,t){if(e==null)return{};var n={};for(var r in e)if({}.hasOwnProperty.call(e,r)){if(t.indexOf(r)!==-1)continue;n[r]=e[r]}return n}function Nb(e,t){return bb(e)||Cb(e,t)||Ab(e,t)||zb()}function Tb(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t);if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function Lb(e){var t=Tb(e,"string");return typeof t=="symbol"?t:t+""}function Ab(e,t){if(e){if(typeof e=="string")return hp(e,t);var n={}.toString.call(e).slice(8,-1);return n==="Object"&&e.constructor&&(n=e.constructor.name),n==="Map"||n==="Set"?Array.from(e):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?hp(e,t):void 0}}function Rb(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function xp(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function _p(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?xp(Object(n),!0).forEach(function(r){Rb(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):xp(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function jb(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(r){return t.reduceRight(function(i,a){return a(i)},r)}}function ti(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function fo(e){return{}.toString.call(e).includes("Object")}function Ib(e){return!Object.keys(e).length}function Fi(e){return typeof e=="function"}function Db(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function Fb(e,t){return fo(t)||wn("changeType"),Object.keys(t).some(function(n){return!Db(e,n)})&&wn("changeField"),t}function Mb(e){Fi(e)||wn("selectorType")}function Ob(e){Fi(e)||fo(e)||wn("handlerType"),fo(e)&&Object.values(e).some(function(t){return!Fi(t)})&&wn("handlersType")}function qb(e){e||wn("initialIsRequired"),fo(e)||wn("initialType"),Ib(e)&&wn("initialContent")}function Wb(e,t){throw new Error(e[t]||e.default)}var Bb={initialIsRequired:"initial state is required",initialType:"initial state should be an object",initialContent:"initial state shouldn't be an empty object",handlerType:"handler should be an object or a function",handlersType:"all handlers should be a functions",selectorType:"selector should be a function",changeType:"provided value of changes should be an object",changeField:'it seams you want to change a field in the state which is not specified in the "initial" state',default:"an unknown error accured in `state-local` package"},wn=ti(Wb)(Bb),_a={changes:Fb,selector:Mb,handler:Ob,initial:qb};function Ub(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};_a.initial(e),_a.handler(t);var n={current:e},r=ti($b)(n,t),i=ti(Hb)(n),a=ti(_a.changes)(e),o=ti(Vb)(n);function s(){var u=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(d){return d};return _a.selector(u),u(n.current)}function l(u){jb(r,i,a,o)(u)}return[s,l]}function Vb(e,t){return Fi(t)?t(e.current):t}function Hb(e,t){return e.current=_p(_p({},e.current),t),t}function $b(e,t,n){return Fi(t)?t(e.current):Object.keys(n).forEach(function(r){var i;return(i=t[r])===null||i===void 0?void 0:i.call(t,e.current[r])}),n}var Qb={create:Ub},Xb={paths:{vs:"https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs"}};function Kb(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function Yb(e){return{}.toString.call(e).includes("Object")}function Gb(e){return e||vp("configIsRequired"),Yb(e)||vp("configType"),e.urls?(Zb(),{paths:{vs:e.urls.monacoBase}}):e}function Zb(){console.warn($h.deprecation)}function Jb(e,t){throw new Error(e[t]||e.default)}var $h={configIsRequired:"the configuration object is required",configType:"the configuration object should be an object",default:"an unknown error accured in `@monaco-editor/loader` package",deprecation:`Deprecation warning!
    You are using deprecated way of configuration.

    Instead of using
      monaco.config({ urls: { monacoBase: '...' } })
    use
      monaco.config({ paths: { vs: '...' } })

    For more please check the link https://github.com/suren-atoyan/monaco-loader#config
  `},vp=Kb(Jb)($h),e2={config:Gb},t2=function(){for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return function(i){return n.reduceRight(function(a,o){return o(a)},i)}};function Qh(e,t){return Object.keys(t).forEach(function(n){t[n]instanceof Object&&e[n]&&Object.assign(t[n],Qh(e[n],t[n]))}),yp(yp({},e),t)}var n2={type:"cancelation",msg:"operation is manually canceled"};function Ns(e){var t=!1,n=new Promise(function(r,i){e.then(function(a){return t?i(n2):r(a)}),e.catch(i)});return n.cancel=function(){return t=!0},n}var r2=["monaco"],i2=Qb.create({config:Xb,isInitialized:!1,resolve:null,reject:null,monaco:null}),Xh=Nb(i2,2),Yi=Xh[0],Wo=Xh[1];function a2(e){var t=e2.config(e),n=t.monaco,r=Eb(t,r2);Wo(function(i){return{config:Qh(i.config,r),monaco:n}})}function o2(){var e=Yi(function(t){var n=t.monaco,r=t.isInitialized,i=t.resolve;return{monaco:n,isInitialized:r,resolve:i}});if(!e.isInitialized){if(Wo({isInitialized:!0}),e.monaco)return e.resolve(e.monaco),Ns(Ts);if(window.monaco&&window.monaco.editor)return Kh(window.monaco),e.resolve(window.monaco),Ns(Ts);t2(s2,u2)(c2)}return Ns(Ts)}function s2(e){return document.body.appendChild(e)}function l2(e){var t=document.createElement("script");return e&&(t.src=e),t}function u2(e){var t=Yi(function(r){var i=r.config,a=r.reject;return{config:i,reject:a}}),n=l2("".concat(t.config.paths.vs,"/loader.js"));return n.onload=function(){return e()},n.onerror=t.reject,n}function c2(){var e=Yi(function(n){var r=n.config,i=n.resolve,a=n.reject;return{config:r,resolve:i,reject:a}}),t=window.require;t.config(e.config),t(["vs/editor/editor.main"],function(n){var r=n.m||n;Kh(r),e.resolve(r)},function(n){e.reject(n)})}function Kh(e){Yi().monaco||Wo({monaco:e})}function d2(){return Yi(function(e){var t=e.monaco;return t})}var Ts=new Promise(function(e,t){return Wo({resolve:e,reject:t})}),Yh={config:a2,init:o2,__getMonacoInstance:d2},p2={wrapper:{display:"flex",position:"relative",textAlign:"initial"},fullWidth:{width:"100%"},hide:{display:"none"}},Ls=p2,f2={container:{display:"flex",height:"100%",width:"100%",justifyContent:"center",alignItems:"center"}},m2=f2;function h2({children:e}){return Ft.createElement("div",{style:m2.container},e)}var g2=h2,y2=g2;function x2({width:e,height:t,isEditorReady:n,loading:r,_ref:i,className:a,wrapperProps:o}){return Ft.createElement("section",{style:{...Ls.wrapper,width:e,height:t},...o},!n&&Ft.createElement(y2,null,r),Ft.createElement("div",{ref:i,style:{...Ls.fullWidth,...!n&&Ls.hide},className:a}))}var _2=x2,Gh=z.memo(_2);function v2(e){z.useEffect(e,[])}var Zh=v2;function w2(e,t,n=!0){let r=z.useRef(!0);z.useEffect(r.current||!n?()=>{r.current=!1}:e,t)}var rt=w2;function mi(){}function fr(e,t,n,r){return k2(e,r)||b2(e,t,n,r)}function k2(e,t){return e.editor.getModel(Jh(e,t))}function b2(e,t,n,r){return e.editor.createModel(t,n,r?Jh(e,r):void 0)}function Jh(e,t){return e.Uri.parse(t)}function S2({original:e,modified:t,language:n,originalLanguage:r,modifiedLanguage:i,originalModelPath:a,modifiedModelPath:o,keepCurrentOriginalModel:s=!1,keepCurrentModifiedModel:l=!1,theme:u="light",loading:d="Loading...",options:c={},height:p="100%",width:f="100%",className:y,wrapperProps:_={},beforeMount:S=mi,onMount:m=mi}){let[h,g]=z.useState(!1),[C,E]=z.useState(!0),b=z.useRef(null),P=z.useRef(null),T=z.useRef(null),D=z.useRef(m),k=z.useRef(S),I=z.useRef(!1);Zh(()=>{let F=Yh.init();return F.then(X=>(P.current=X)&&E(!1)).catch(X=>(X==null?void 0:X.type)!=="cancelation"&&console.error("Monaco initialization: error:",X)),()=>b.current?te():F.cancel()}),rt(()=>{if(b.current&&P.current){let F=b.current.getOriginalEditor(),X=fr(P.current,e||"",r||n||"text",a||"");X!==F.getModel()&&F.setModel(X)}},[a],h),rt(()=>{if(b.current&&P.current){let F=b.current.getModifiedEditor(),X=fr(P.current,t||"",i||n||"text",o||"");X!==F.getModel()&&F.setModel(X)}},[o],h),rt(()=>{let F=b.current.getModifiedEditor();F.getOption(P.current.editor.EditorOption.readOnly)?F.setValue(t||""):t!==F.getValue()&&(F.executeEdits("",[{range:F.getModel().getFullModelRange(),text:t||"",forceMoveMarkers:!0}]),F.pushUndoStop())},[t],h),rt(()=>{var F,X;(X=(F=b.current)==null?void 0:F.getModel())==null||X.original.setValue(e||"")},[e],h),rt(()=>{let{original:F,modified:X}=b.current.getModel();P.current.editor.setModelLanguage(F,r||n||"text"),P.current.editor.setModelLanguage(X,i||n||"text")},[n,r,i],h),rt(()=>{var F;(F=P.current)==null||F.editor.setTheme(u)},[u],h),rt(()=>{var F;(F=b.current)==null||F.updateOptions(c)},[c],h);let W=z.useCallback(()=>{var re;if(!P.current)return;k.current(P.current);let F=fr(P.current,e||"",r||n||"text",a||""),X=fr(P.current,t||"",i||n||"text",o||"");(re=b.current)==null||re.setModel({original:F,modified:X})},[n,t,i,e,r,a,o]),K=z.useCallback(()=>{var F;!I.current&&T.current&&(b.current=P.current.editor.createDiffEditor(T.current,{automaticLayout:!0,...c}),W(),(F=P.current)==null||F.editor.setTheme(u),g(!0),I.current=!0)},[c,u,W]);z.useEffect(()=>{h&&D.current(b.current,P.current)},[h]),z.useEffect(()=>{!C&&!h&&K()},[C,h,K]);function te(){var X,re,j,V;let F=(X=b.current)==null?void 0:X.getModel();s||((re=F==null?void 0:F.original)==null||re.dispose()),l||((j=F==null?void 0:F.modified)==null||j.dispose()),(V=b.current)==null||V.dispose()}return Ft.createElement(Gh,{width:f,height:p,isEditorReady:h,loading:d,_ref:T,className:y,wrapperProps:_})}var C2=S2;z.memo(C2);function z2(e){let t=z.useRef();return z.useEffect(()=>{t.current=e},[e]),t.current}var E2=z2,va=new Map;function P2({defaultValue:e,defaultLanguage:t,defaultPath:n,value:r,language:i,path:a,theme:o="light",line:s,loading:l="Loading...",options:u={},overrideServices:d={},saveViewState:c=!0,keepCurrentModel:p=!1,width:f="100%",height:y="100%",className:_,wrapperProps:S={},beforeMount:m=mi,onMount:h=mi,onChange:g,onValidate:C=mi}){let[E,b]=z.useState(!1),[P,T]=z.useState(!0),D=z.useRef(null),k=z.useRef(null),I=z.useRef(null),W=z.useRef(h),K=z.useRef(m),te=z.useRef(),F=z.useRef(r),X=E2(a),re=z.useRef(!1),j=z.useRef(!1);Zh(()=>{let B=Yh.init();return B.then($=>(D.current=$)&&T(!1)).catch($=>($==null?void 0:$.type)!=="cancelation"&&console.error("Monaco initialization: error:",$)),()=>k.current?v():B.cancel()}),rt(()=>{var $,w,ie,ke;let B=fr(D.current,e||r||"",t||i||"",a||n||"");B!==(($=k.current)==null?void 0:$.getModel())&&(c&&va.set(X,(w=k.current)==null?void 0:w.saveViewState()),(ie=k.current)==null||ie.setModel(B),c&&((ke=k.current)==null||ke.restoreViewState(va.get(a))))},[a],E),rt(()=>{var B;(B=k.current)==null||B.updateOptions(u)},[u],E),rt(()=>{!k.current||r===void 0||(k.current.getOption(D.current.editor.EditorOption.readOnly)?k.current.setValue(r):r!==k.current.getValue()&&(j.current=!0,k.current.executeEdits("",[{range:k.current.getModel().getFullModelRange(),text:r,forceMoveMarkers:!0}]),k.current.pushUndoStop(),j.current=!1))},[r],E),rt(()=>{var $,w;let B=($=k.current)==null?void 0:$.getModel();B&&i&&((w=D.current)==null||w.editor.setModelLanguage(B,i))},[i],E),rt(()=>{var B;s!==void 0&&((B=k.current)==null||B.revealLine(s))},[s],E),rt(()=>{var B;(B=D.current)==null||B.editor.setTheme(o)},[o],E);let V=z.useCallback(()=>{var B;if(!(!I.current||!D.current)&&!re.current){K.current(D.current);let $=a||n,w=fr(D.current,r||e||"",t||i||"",$||"");k.current=(B=D.current)==null?void 0:B.editor.create(I.current,{model:w,automaticLayout:!0,...u},d),c&&k.current.restoreViewState(va.get($)),D.current.editor.setTheme(o),s!==void 0&&k.current.revealLine(s),b(!0),re.current=!0}},[e,t,n,r,i,a,u,d,c,o,s]);z.useEffect(()=>{E&&W.current(k.current,D.current)},[E]),z.useEffect(()=>{!P&&!E&&V()},[P,E,V]),F.current=r,z.useEffect(()=>{var B,$;E&&g&&((B=te.current)==null||B.dispose(),te.current=($=k.current)==null?void 0:$.onDidChangeModelContent(w=>{j.current||g(k.current.getValue(),w)}))},[E,g]),z.useEffect(()=>{if(E){let B=D.current.editor.onDidChangeMarkers($=>{var ie;let w=(ie=k.current.getModel())==null?void 0:ie.uri;if(w&&$.find(ke=>ke.path===w.path)){let ke=D.current.editor.getModelMarkers({resource:w});C==null||C(ke)}});return()=>{B==null||B.dispose()}}return()=>{}},[E,C]);function v(){var B,$;(B=te.current)==null||B.dispose(),p?c&&va.set(a,k.current.saveViewState()):($=k.current.getModel())==null||$.dispose(),k.current.dispose()}return Ft.createElement(Gh,{width:f,height:y,isEditorReady:E,loading:l,_ref:I,className:_,wrapperProps:S})}var N2=P2,T2=z.memo(N2),L2=T2;function eg({value:e,onChange:t,height:n="400px"}){const r=z.useCallback(i=>{t(i||"")},[t]);return x.jsx("div",{className:"h-full border border-gray-200 rounded-lg overflow-hidden shadow-sm",children:x.jsx(L2,{height:n,defaultLanguage:"python",theme:"light",value:e,onChange:r,options:{minimap:{enabled:!1},fontSize:14,lineNumbers:"on",scrollBeyondLastLine:!1,automaticLayout:!0,tabSize:4,wordWrap:"on",padding:{top:16,bottom:16},scrollbar:{verticalScrollbarSize:8,horizontalScrollbarSize:8}}})})}function tg({output:e,isLoading:t=!1}){const n=z.useRef(null);return z.useEffect(()=>{var r;(r=n.current)==null||r.scrollIntoView({behavior:"smooth"})},[e]),x.jsxs("div",{className:"bg-white border border-gray-200 rounded-lg h-full flex flex-col shadow-sm",children:[x.jsxs("div",{className:"flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-gray-50",children:[x.jsx("span",{className:"text-sm font-medium text-gray-700",children:"Console Output"}),t&&x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-primary-500 rounded-full animate-pulse"}),x.jsx("span",{className:"text-xs text-gray-500",children:"Running..."})]})]}),x.jsxs("div",{className:"flex-1 overflow-auto p-4 font-mono text-sm bg-gray-900 rounded-b-lg",children:[e.length===0?x.jsx("div",{className:"text-gray-500 italic",children:"Output will appear here when you run your code..."}):e.map((r,i)=>x.jsx("div",{className:`py-0.5 ${r.startsWith("[Error]")?"text-red-400":r.includes("PASSED")?"text-green-400":r.includes("FAILED")?"text-red-400":"text-gray-300"}`,children:r.startsWith("Test")?x.jsxs("span",{children:[r.includes("PASSED")?"✓ ":r.includes("FAILED")?"✗ ":"",r]}):x.jsxs("span",{children:["> ",r]})},i)),x.jsx("div",{ref:n})]})]})}function wp(e,t){const n=String(e);if(typeof t!="string")throw new TypeError("Expected character");let r=0,i=n.indexOf(t);for(;i!==-1;)r++,i=n.indexOf(t,i+t.length);return r}function A2(e){if(typeof e!="string")throw new TypeError("Expected a string");return e.replace(/[|\\{}()[\]^$+*?.]/g,"\\$&").replace(/-/g,"\\x2d")}function R2(e,t,n){const i=Oo((n||{}).ignore||[]),a=j2(t);let o=-1;for(;++o<a.length;)Dh(e,"text",s);function s(u,d){let c=-1,p;for(;++c<d.length;){const f=d[c],y=p?p.children:void 0;if(i(f,y?y.indexOf(f):void 0,p))return;p=f}if(p)return l(u,d)}function l(u,d){const c=d[d.length-1],p=a[o][0],f=a[o][1];let y=0;const S=c.children.indexOf(u);let m=!1,h=[];p.lastIndex=0;let g=p.exec(u.value);for(;g;){const C=g.index,E={index:g.index,input:g.input,stack:[...d,u]};let b=f(...g,E);if(typeof b=="string"&&(b=b.length>0?{type:"text",value:b}:void 0),b===!1?p.lastIndex=C+1:(y!==C&&h.push({type:"text",value:u.value.slice(y,C)}),Array.isArray(b)?h.push(...b):b&&h.push(b),y=C+g[0].length,m=!0),!p.global)break;g=p.exec(u.value)}return m?(y<u.value.length&&h.push({type:"text",value:u.value.slice(y)}),c.children.splice(S,1,...h)):h=[u],S+h.length}}function j2(e){const t=[];if(!Array.isArray(e))throw new TypeError("Expected find and replace tuple or list of tuples");const n=!e[0]||Array.isArray(e[0])?e:[e];let r=-1;for(;++r<n.length;){const i=n[r];t.push([I2(i[0]),D2(i[1])])}return t}function I2(e){return typeof e=="string"?new RegExp(A2(e),"g"):e}function D2(e){return typeof e=="function"?e:function(){return e}}const As="phrasing",Rs=["autolink","link","image","label"];function F2(){return{transforms:[V2],enter:{literalAutolink:O2,literalAutolinkEmail:js,literalAutolinkHttp:js,literalAutolinkWww:js},exit:{literalAutolink:U2,literalAutolinkEmail:B2,literalAutolinkHttp:q2,literalAutolinkWww:W2}}}function M2(){return{unsafe:[{character:"@",before:"[+\\-.\\w]",after:"[\\-.\\w]",inConstruct:As,notInConstruct:Rs},{character:".",before:"[Ww]",after:"[\\-.\\w]",inConstruct:As,notInConstruct:Rs},{character:":",before:"[ps]",after:"\\/",inConstruct:As,notInConstruct:Rs}]}}function O2(e){this.enter({type:"link",title:null,url:"",children:[]},e)}function js(e){this.config.enter.autolinkProtocol.call(this,e)}function q2(e){this.config.exit.autolinkProtocol.call(this,e)}function W2(e){this.config.exit.data.call(this,e);const t=this.stack[this.stack.length-1];t.type,t.url="http://"+this.sliceSerialize(e)}function B2(e){this.config.exit.autolinkEmail.call(this,e)}function U2(e){this.exit(e)}function V2(e){R2(e,[[/(https?:\/\/|www(?=\.))([-.\w]+)([^ \t\r\n]*)/gi,H2],[new RegExp("(?<=^|\\s|\\p{P}|\\p{S})([-.\\w+]+)@([-\\w]+(?:\\.[-\\w]+)+)","gu"),$2]],{ignore:["link","linkReference"]})}function H2(e,t,n,r,i){let a="";if(!ng(i)||(/^w/i.test(t)&&(n=t+n,t="",a="http://"),!Q2(n)))return!1;const o=X2(n+r);if(!o[0])return!1;const s={type:"link",title:null,url:a+t+o[0],children:[{type:"text",value:t+o[0]}]};return o[1]?[s,{type:"text",value:o[1]}]:s}function $2(e,t,n,r){return!ng(r,!0)||/[-\d_]$/.test(n)?!1:{type:"link",title:null,url:"mailto:"+t+"@"+n,children:[{type:"text",value:t+"@"+n}]}}function Q2(e){const t=e.split(".");return!(t.length<2||t[t.length-1]&&(/_/.test(t[t.length-1])||!/[a-zA-Z\d]/.test(t[t.length-1]))||t[t.length-2]&&(/_/.test(t[t.length-2])||!/[a-zA-Z\d]/.test(t[t.length-2])))}function X2(e){const t=/[!"&'),.:;<>?\]}]+$/.exec(e);if(!t)return[e,void 0];e=e.slice(0,t.index);let n=t[0],r=n.indexOf(")");const i=wp(e,"(");let a=wp(e,")");for(;r!==-1&&i>a;)e+=n.slice(0,r+1),n=n.slice(r+1),r=n.indexOf(")"),a++;return[e,n]}function ng(e,t){const n=e.input.charCodeAt(e.index-1);return(e.index===0||Qn(n)||Do(n))&&(!t||n!==47)}rg.peek=rS;function K2(){this.buffer()}function Y2(e){this.enter({type:"footnoteReference",identifier:"",label:""},e)}function G2(){this.buffer()}function Z2(e){this.enter({type:"footnoteDefinition",identifier:"",label:"",children:[]},e)}function J2(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Tt(this.sliceSerialize(e)).toLowerCase(),n.label=t}function eS(e){this.exit(e)}function tS(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Tt(this.sliceSerialize(e)).toLowerCase(),n.label=t}function nS(e){this.exit(e)}function rS(){return"["}function rg(e,t,n,r){const i=n.createTracker(r);let a=i.move("[^");const o=n.enter("footnoteReference"),s=n.enter("reference");return a+=i.move(n.safe(n.associationId(e),{after:"]",before:a})),s(),o(),a+=i.move("]"),a}function iS(){return{enter:{gfmFootnoteCallString:K2,gfmFootnoteCall:Y2,gfmFootnoteDefinitionLabelString:G2,gfmFootnoteDefinition:Z2},exit:{gfmFootnoteCallString:J2,gfmFootnoteCall:eS,gfmFootnoteDefinitionLabelString:tS,gfmFootnoteDefinition:nS}}}function aS(e){let t=!1;return e&&e.firstLineBlank&&(t=!0),{handlers:{footnoteDefinition:n,footnoteReference:rg},unsafe:[{character:"[",inConstruct:["label","phrasing","reference"]}]};function n(r,i,a,o){const s=a.createTracker(o);let l=s.move("[^");const u=a.enter("footnoteDefinition"),d=a.enter("label");return l+=s.move(a.safe(a.associationId(r),{before:l,after:"]"})),d(),l+=s.move("]:"),r.children&&r.children.length>0&&(s.shift(4),l+=s.move((t?`
`:" ")+a.indentLines(a.containerFlow(r,s.current()),t?ig:oS))),u(),l}}function oS(e,t,n){return t===0?e:ig(e,t,n)}function ig(e,t,n){return(n?"":"    ")+e}const sS=["autolink","destinationLiteral","destinationRaw","reference","titleQuote","titleApostrophe"];ag.peek=pS;function lS(){return{canContainEols:["delete"],enter:{strikethrough:cS},exit:{strikethrough:dS}}}function uS(){return{unsafe:[{character:"~",inConstruct:"phrasing",notInConstruct:sS}],handlers:{delete:ag}}}function cS(e){this.enter({type:"delete",children:[]},e)}function dS(e){this.exit(e)}function ag(e,t,n,r){const i=n.createTracker(r),a=n.enter("strikethrough");let o=i.move("~~");return o+=n.containerPhrasing(e,{...i.current(),before:o,after:"~"}),o+=i.move("~~"),a(),o}function pS(){return"~"}function fS(e){return e.length}function mS(e,t){const n=t||{},r=(n.align||[]).concat(),i=n.stringLength||fS,a=[],o=[],s=[],l=[];let u=0,d=-1;for(;++d<e.length;){const _=[],S=[];let m=-1;for(e[d].length>u&&(u=e[d].length);++m<e[d].length;){const h=hS(e[d][m]);if(n.alignDelimiters!==!1){const g=i(h);S[m]=g,(l[m]===void 0||g>l[m])&&(l[m]=g)}_.push(h)}o[d]=_,s[d]=S}let c=-1;if(typeof r=="object"&&"length"in r)for(;++c<u;)a[c]=kp(r[c]);else{const _=kp(r);for(;++c<u;)a[c]=_}c=-1;const p=[],f=[];for(;++c<u;){const _=a[c];let S="",m="";_===99?(S=":",m=":"):_===108?S=":":_===114&&(m=":");let h=n.alignDelimiters===!1?1:Math.max(1,l[c]-S.length-m.length);const g=S+"-".repeat(h)+m;n.alignDelimiters!==!1&&(h=S.length+h+m.length,h>l[c]&&(l[c]=h),f[c]=h),p[c]=g}o.splice(1,0,p),s.splice(1,0,f),d=-1;const y=[];for(;++d<o.length;){const _=o[d],S=s[d];c=-1;const m=[];for(;++c<u;){const h=_[c]||"";let g="",C="";if(n.alignDelimiters!==!1){const E=l[c]-(S[c]||0),b=a[c];b===114?g=" ".repeat(E):b===99?E%2?(g=" ".repeat(E/2+.5),C=" ".repeat(E/2-.5)):(g=" ".repeat(E/2),C=g):C=" ".repeat(E)}n.delimiterStart!==!1&&!c&&m.push("|"),n.padding!==!1&&!(n.alignDelimiters===!1&&h==="")&&(n.delimiterStart!==!1||c)&&m.push(" "),n.alignDelimiters!==!1&&m.push(g),m.push(h),n.alignDelimiters!==!1&&m.push(C),n.padding!==!1&&m.push(" "),(n.delimiterEnd!==!1||c!==u-1)&&m.push("|")}y.push(n.delimiterEnd===!1?m.join("").replace(/ +$/,""):m.join(""))}return y.join(`
`)}function hS(e){return e==null?"":String(e)}function kp(e){const t=typeof e=="string"?e.codePointAt(0):0;return t===67||t===99?99:t===76||t===108?108:t===82||t===114?114:0}function gS(e,t,n,r){const i=n.enter("blockquote"),a=n.createTracker(r);a.move("> "),a.shift(2);const o=n.indentLines(n.containerFlow(e,a.current()),yS);return i(),o}function yS(e,t,n){return">"+(n?"":" ")+e}function xS(e,t){return bp(e,t.inConstruct,!0)&&!bp(e,t.notInConstruct,!1)}function bp(e,t,n){if(typeof t=="string"&&(t=[t]),!t||t.length===0)return n;let r=-1;for(;++r<t.length;)if(e.includes(t[r]))return!0;return!1}function Sp(e,t,n,r){let i=-1;for(;++i<n.unsafe.length;)if(n.unsafe[i].character===`
`&&xS(n.stack,n.unsafe[i]))return/[ \t]/.test(r.before)?"":" ";return`\\
`}function _S(e,t){const n=String(e);let r=n.indexOf(t),i=r,a=0,o=0;if(typeof t!="string")throw new TypeError("Expected substring");for(;r!==-1;)r===i?++a>o&&(o=a):a=1,i=r+t.length,r=n.indexOf(t,i);return o}function vS(e,t){return!!(t.options.fences===!1&&e.value&&!e.lang&&/[^ \r\n]/.test(e.value)&&!/^[\t ]*(?:[\r\n]|$)|(?:^|[\r\n])[\t ]*$/.test(e.value))}function wS(e){const t=e.options.fence||"`";if(t!=="`"&&t!=="~")throw new Error("Cannot serialize code with `"+t+"` for `options.fence`, expected `` ` `` or `~`");return t}function kS(e,t,n,r){const i=wS(n),a=e.value||"",o=i==="`"?"GraveAccent":"Tilde";if(vS(e,n)){const c=n.enter("codeIndented"),p=n.indentLines(a,bS);return c(),p}const s=n.createTracker(r),l=i.repeat(Math.max(_S(a,i)+1,3)),u=n.enter("codeFenced");let d=s.move(l);if(e.lang){const c=n.enter(`codeFencedLang${o}`);d+=s.move(n.safe(e.lang,{before:d,after:" ",encode:["`"],...s.current()})),c()}if(e.lang&&e.meta){const c=n.enter(`codeFencedMeta${o}`);d+=s.move(" "),d+=s.move(n.safe(e.meta,{before:d,after:`
`,encode:["`"],...s.current()})),c()}return d+=s.move(`
`),a&&(d+=s.move(a+`
`)),d+=s.move(l),u(),d}function bS(e,t,n){return(n?"":"    ")+e}function rc(e){const t=e.options.quote||'"';if(t!=='"'&&t!=="'")throw new Error("Cannot serialize title with `"+t+"` for `options.quote`, expected `\"`, or `'`");return t}function SS(e,t,n,r){const i=rc(n),a=i==='"'?"Quote":"Apostrophe",o=n.enter("definition");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("[");return u+=l.move(n.safe(n.associationId(e),{before:u,after:"]",...l.current()})),u+=l.move("]: "),s(),!e.url||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":`
`,...l.current()}))),s(),e.title&&(s=n.enter(`title${a}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),o(),u}function CS(e){const t=e.options.emphasis||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize emphasis with `"+t+"` for `options.emphasis`, expected `*`, or `_`");return t}function Mi(e){return"&#x"+e.toString(16).toUpperCase()+";"}function mo(e,t,n){const r=Tr(e),i=Tr(t);return r===void 0?i===void 0?n==="_"?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!0}:r===1?i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!1}:{inside:!1,outside:!1}}og.peek=zS;function og(e,t,n,r){const i=CS(n),a=n.enter("emphasis"),o=n.createTracker(r),s=o.move(i);let l=o.move(n.containerPhrasing(e,{after:i,before:s,...o.current()}));const u=l.charCodeAt(0),d=mo(r.before.charCodeAt(r.before.length-1),u,i);d.inside&&(l=Mi(u)+l.slice(1));const c=l.charCodeAt(l.length-1),p=mo(r.after.charCodeAt(0),c,i);p.inside&&(l=l.slice(0,-1)+Mi(c));const f=o.move(i);return a(),n.attentionEncodeSurroundingInfo={after:p.outside,before:d.outside},s+l+f}function zS(e,t,n){return n.options.emphasis||"*"}function ES(e,t){let n=!1;return ec(e,function(r){if("value"in r&&/\r?\n|\r/.test(r.value)||r.type==="break")return n=!0,ql}),!!((!e.depth||e.depth<3)&&Qu(e)&&(t.options.setext||n))}function PS(e,t,n,r){const i=Math.max(Math.min(6,e.depth||1),1),a=n.createTracker(r);if(ES(e,n)){const d=n.enter("headingSetext"),c=n.enter("phrasing"),p=n.containerPhrasing(e,{...a.current(),before:`
`,after:`
`});return c(),d(),p+`
`+(i===1?"=":"-").repeat(p.length-(Math.max(p.lastIndexOf("\r"),p.lastIndexOf(`
`))+1))}const o="#".repeat(i),s=n.enter("headingAtx"),l=n.enter("phrasing");a.move(o+" ");let u=n.containerPhrasing(e,{before:"# ",after:`
`,...a.current()});return/^[\t ]/.test(u)&&(u=Mi(u.charCodeAt(0))+u.slice(1)),u=u?o+" "+u:o,n.options.closeAtx&&(u+=" "+o),l(),s(),u}sg.peek=NS;function sg(e){return e.value||""}function NS(){return"<"}lg.peek=TS;function lg(e,t,n,r){const i=rc(n),a=i==='"'?"Quote":"Apostrophe",o=n.enter("image");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("![");return u+=l.move(n.safe(e.alt,{before:u,after:"]",...l.current()})),u+=l.move("]("),s(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":")",...l.current()}))),s(),e.title&&(s=n.enter(`title${a}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),u+=l.move(")"),o(),u}function TS(){return"!"}ug.peek=LS;function ug(e,t,n,r){const i=e.referenceType,a=n.enter("imageReference");let o=n.enter("label");const s=n.createTracker(r);let l=s.move("![");const u=n.safe(e.alt,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),o();const d=n.stack;n.stack=[],o=n.enter("reference");const c=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return o(),n.stack=d,a(),i==="full"||!u||u!==c?l+=s.move(c+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function LS(){return"!"}cg.peek=AS;function cg(e,t,n){let r=e.value||"",i="`",a=-1;for(;new RegExp("(^|[^`])"+i+"([^`]|$)").test(r);)i+="`";for(/[^ \r\n]/.test(r)&&(/^[ \r\n]/.test(r)&&/[ \r\n]$/.test(r)||/^`|`$/.test(r))&&(r=" "+r+" ");++a<n.unsafe.length;){const o=n.unsafe[a],s=n.compilePattern(o);let l;if(o.atBreak)for(;l=s.exec(r);){let u=l.index;r.charCodeAt(u)===10&&r.charCodeAt(u-1)===13&&u--,r=r.slice(0,u)+" "+r.slice(l.index+1)}}return i+r+i}function AS(){return"`"}function dg(e,t){const n=Qu(e);return!!(!t.options.resourceLink&&e.url&&!e.title&&e.children&&e.children.length===1&&e.children[0].type==="text"&&(n===e.url||"mailto:"+n===e.url)&&/^[a-z][a-z+.-]+:/i.test(e.url)&&!/[\0- <>\u007F]/.test(e.url))}pg.peek=RS;function pg(e,t,n,r){const i=rc(n),a=i==='"'?"Quote":"Apostrophe",o=n.createTracker(r);let s,l;if(dg(e,n)){const d=n.stack;n.stack=[],s=n.enter("autolink");let c=o.move("<");return c+=o.move(n.containerPhrasing(e,{before:c,after:">",...o.current()})),c+=o.move(">"),s(),n.stack=d,c}s=n.enter("link"),l=n.enter("label");let u=o.move("[");return u+=o.move(n.containerPhrasing(e,{before:u,after:"](",...o.current()})),u+=o.move("]("),l(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(l=n.enter("destinationLiteral"),u+=o.move("<"),u+=o.move(n.safe(e.url,{before:u,after:">",...o.current()})),u+=o.move(">")):(l=n.enter("destinationRaw"),u+=o.move(n.safe(e.url,{before:u,after:e.title?" ":")",...o.current()}))),l(),e.title&&(l=n.enter(`title${a}`),u+=o.move(" "+i),u+=o.move(n.safe(e.title,{before:u,after:i,...o.current()})),u+=o.move(i),l()),u+=o.move(")"),s(),u}function RS(e,t,n){return dg(e,n)?"<":"["}fg.peek=jS;function fg(e,t,n,r){const i=e.referenceType,a=n.enter("linkReference");let o=n.enter("label");const s=n.createTracker(r);let l=s.move("[");const u=n.containerPhrasing(e,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),o();const d=n.stack;n.stack=[],o=n.enter("reference");const c=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return o(),n.stack=d,a(),i==="full"||!u||u!==c?l+=s.move(c+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function jS(){return"["}function ic(e){const t=e.options.bullet||"*";if(t!=="*"&&t!=="+"&&t!=="-")throw new Error("Cannot serialize items with `"+t+"` for `options.bullet`, expected `*`, `+`, or `-`");return t}function IS(e){const t=ic(e),n=e.options.bulletOther;if(!n)return t==="*"?"-":"*";if(n!=="*"&&n!=="+"&&n!=="-")throw new Error("Cannot serialize items with `"+n+"` for `options.bulletOther`, expected `*`, `+`, or `-`");if(n===t)throw new Error("Expected `bullet` (`"+t+"`) and `bulletOther` (`"+n+"`) to be different");return n}function DS(e){const t=e.options.bulletOrdered||".";if(t!=="."&&t!==")")throw new Error("Cannot serialize items with `"+t+"` for `options.bulletOrdered`, expected `.` or `)`");return t}function mg(e){const t=e.options.rule||"*";if(t!=="*"&&t!=="-"&&t!=="_")throw new Error("Cannot serialize rules with `"+t+"` for `options.rule`, expected `*`, `-`, or `_`");return t}function FS(e,t,n,r){const i=n.enter("list"),a=n.bulletCurrent;let o=e.ordered?DS(n):ic(n);const s=e.ordered?o==="."?")":".":IS(n);let l=t&&n.bulletLastUsed?o===n.bulletLastUsed:!1;if(!e.ordered){const d=e.children?e.children[0]:void 0;if((o==="*"||o==="-")&&d&&(!d.children||!d.children[0])&&n.stack[n.stack.length-1]==="list"&&n.stack[n.stack.length-2]==="listItem"&&n.stack[n.stack.length-3]==="list"&&n.stack[n.stack.length-4]==="listItem"&&n.indexStack[n.indexStack.length-1]===0&&n.indexStack[n.indexStack.length-2]===0&&n.indexStack[n.indexStack.length-3]===0&&(l=!0),mg(n)===o&&d){let c=-1;for(;++c<e.children.length;){const p=e.children[c];if(p&&p.type==="listItem"&&p.children&&p.children[0]&&p.children[0].type==="thematicBreak"){l=!0;break}}}}l&&(o=s),n.bulletCurrent=o;const u=n.containerFlow(e,r);return n.bulletLastUsed=o,n.bulletCurrent=a,i(),u}function MS(e){const t=e.options.listItemIndent||"one";if(t!=="tab"&&t!=="one"&&t!=="mixed")throw new Error("Cannot serialize items with `"+t+"` for `options.listItemIndent`, expected `tab`, `one`, or `mixed`");return t}function OS(e,t,n,r){const i=MS(n);let a=n.bulletCurrent||ic(n);t&&t.type==="list"&&t.ordered&&(a=(typeof t.start=="number"&&t.start>-1?t.start:1)+(n.options.incrementListMarker===!1?0:t.children.indexOf(e))+a);let o=a.length+1;(i==="tab"||i==="mixed"&&(t&&t.type==="list"&&t.spread||e.spread))&&(o=Math.ceil(o/4)*4);const s=n.createTracker(r);s.move(a+" ".repeat(o-a.length)),s.shift(o);const l=n.enter("listItem"),u=n.indentLines(n.containerFlow(e,s.current()),d);return l(),u;function d(c,p,f){return p?(f?"":" ".repeat(o))+c:(f?a:a+" ".repeat(o-a.length))+c}}function qS(e,t,n,r){const i=n.enter("paragraph"),a=n.enter("phrasing"),o=n.containerPhrasing(e,r);return a(),i(),o}const WS=Oo(["break","delete","emphasis","footnote","footnoteReference","image","imageReference","inlineCode","inlineMath","link","linkReference","mdxJsxTextElement","mdxTextExpression","strong","text","textDirective"]);function BS(e,t,n,r){return(e.children.some(function(o){return WS(o)})?n.containerPhrasing:n.containerFlow).call(n,e,r)}function US(e){const t=e.options.strong||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize strong with `"+t+"` for `options.strong`, expected `*`, or `_`");return t}hg.peek=VS;function hg(e,t,n,r){const i=US(n),a=n.enter("strong"),o=n.createTracker(r),s=o.move(i+i);let l=o.move(n.containerPhrasing(e,{after:i,before:s,...o.current()}));const u=l.charCodeAt(0),d=mo(r.before.charCodeAt(r.before.length-1),u,i);d.inside&&(l=Mi(u)+l.slice(1));const c=l.charCodeAt(l.length-1),p=mo(r.after.charCodeAt(0),c,i);p.inside&&(l=l.slice(0,-1)+Mi(c));const f=o.move(i+i);return a(),n.attentionEncodeSurroundingInfo={after:p.outside,before:d.outside},s+l+f}function VS(e,t,n){return n.options.strong||"*"}function HS(e,t,n,r){return n.safe(e.value,r)}function $S(e){const t=e.options.ruleRepetition||3;if(t<3)throw new Error("Cannot serialize rules with repetition `"+t+"` for `options.ruleRepetition`, expected `3` or more");return t}function QS(e,t,n){const r=(mg(n)+(n.options.ruleSpaces?" ":"")).repeat($S(n));return n.options.ruleSpaces?r.slice(0,-1):r}const gg={blockquote:gS,break:Sp,code:kS,definition:SS,emphasis:og,hardBreak:Sp,heading:PS,html:sg,image:lg,imageReference:ug,inlineCode:cg,link:pg,linkReference:fg,list:FS,listItem:OS,paragraph:qS,root:BS,strong:hg,text:HS,thematicBreak:QS};function XS(){return{enter:{table:KS,tableData:Cp,tableHeader:Cp,tableRow:GS},exit:{codeText:ZS,table:YS,tableData:Is,tableHeader:Is,tableRow:Is}}}function KS(e){const t=e._align;this.enter({type:"table",align:t.map(function(n){return n==="none"?null:n}),children:[]},e),this.data.inTable=!0}function YS(e){this.exit(e),this.data.inTable=void 0}function GS(e){this.enter({type:"tableRow",children:[]},e)}function Is(e){this.exit(e)}function Cp(e){this.enter({type:"tableCell",children:[]},e)}function ZS(e){let t=this.resume();this.data.inTable&&(t=t.replace(/\\([\\|])/g,JS));const n=this.stack[this.stack.length-1];n.type,n.value=t,this.exit(e)}function JS(e,t){return t==="|"?t:e}function eC(e){const t=e||{},n=t.tableCellPadding,r=t.tablePipeAlign,i=t.stringLength,a=n?" ":"|";return{unsafe:[{character:"\r",inConstruct:"tableCell"},{character:`
`,inConstruct:"tableCell"},{atBreak:!0,character:"|",after:"[	 :-]"},{character:"|",inConstruct:"tableCell"},{atBreak:!0,character:":",after:"-"},{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{inlineCode:p,table:o,tableCell:l,tableRow:s}};function o(f,y,_,S){return u(d(f,_,S),f.align)}function s(f,y,_,S){const m=c(f,_,S),h=u([m]);return h.slice(0,h.indexOf(`
`))}function l(f,y,_,S){const m=_.enter("tableCell"),h=_.enter("phrasing"),g=_.containerPhrasing(f,{...S,before:a,after:a});return h(),m(),g}function u(f,y){return mS(f,{align:y,alignDelimiters:r,padding:n,stringLength:i})}function d(f,y,_){const S=f.children;let m=-1;const h=[],g=y.enter("table");for(;++m<S.length;)h[m]=c(S[m],y,_);return g(),h}function c(f,y,_){const S=f.children;let m=-1;const h=[],g=y.enter("tableRow");for(;++m<S.length;)h[m]=l(S[m],f,y,_);return g(),h}function p(f,y,_){let S=gg.inlineCode(f,y,_);return _.stack.includes("tableCell")&&(S=S.replace(/\|/g,"\\$&")),S}}function tC(){return{exit:{taskListCheckValueChecked:zp,taskListCheckValueUnchecked:zp,paragraph:rC}}}function nC(){return{unsafe:[{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{listItem:iC}}}function zp(e){const t=this.stack[this.stack.length-2];t.type,t.checked=e.type==="taskListCheckValueChecked"}function rC(e){const t=this.stack[this.stack.length-2];if(t&&t.type==="listItem"&&typeof t.checked=="boolean"){const n=this.stack[this.stack.length-1];n.type;const r=n.children[0];if(r&&r.type==="text"){const i=t.children;let a=-1,o;for(;++a<i.length;){const s=i[a];if(s.type==="paragraph"){o=s;break}}o===n&&(r.value=r.value.slice(1),r.value.length===0?n.children.shift():n.position&&r.position&&typeof r.position.start.offset=="number"&&(r.position.start.column++,r.position.start.offset++,n.position.start=Object.assign({},r.position.start)))}}this.exit(e)}function iC(e,t,n,r){const i=e.children[0],a=typeof e.checked=="boolean"&&i&&i.type==="paragraph",o="["+(e.checked?"x":" ")+"] ",s=n.createTracker(r);a&&s.move(o);let l=gg.listItem(e,t,n,{...r,...s.current()});return a&&(l=l.replace(/^(?:[*+-]|\d+\.)([\r\n]| {1,3})/,u)),l;function u(d){return d+o}}function aC(){return[F2(),iS(),lS(),XS(),tC()]}function oC(e){return{extensions:[M2(),aS(e),uS(),eC(e),nC()]}}const sC={tokenize:fC,partial:!0},yg={tokenize:mC,partial:!0},xg={tokenize:hC,partial:!0},_g={tokenize:gC,partial:!0},lC={tokenize:yC,partial:!0},vg={name:"wwwAutolink",tokenize:dC,previous:kg},wg={name:"protocolAutolink",tokenize:pC,previous:bg},tn={name:"emailAutolink",tokenize:cC,previous:Sg},Wt={};function uC(){return{text:Wt}}let Rn=48;for(;Rn<123;)Wt[Rn]=tn,Rn++,Rn===58?Rn=65:Rn===91&&(Rn=97);Wt[43]=tn;Wt[45]=tn;Wt[46]=tn;Wt[95]=tn;Wt[72]=[tn,wg];Wt[104]=[tn,wg];Wt[87]=[tn,vg];Wt[119]=[tn,vg];function cC(e,t,n){const r=this;let i,a;return o;function o(c){return!Vl(c)||!Sg.call(r,r.previous)||ac(r.events)?n(c):(e.enter("literalAutolink"),e.enter("literalAutolinkEmail"),s(c))}function s(c){return Vl(c)?(e.consume(c),s):c===64?(e.consume(c),l):n(c)}function l(c){return c===46?e.check(lC,d,u)(c):c===45||c===95||Me(c)?(a=!0,e.consume(c),l):d(c)}function u(c){return e.consume(c),i=!0,l}function d(c){return a&&i&&Ue(r.previous)?(e.exit("literalAutolinkEmail"),e.exit("literalAutolink"),t(c)):n(c)}}function dC(e,t,n){const r=this;return i;function i(o){return o!==87&&o!==119||!kg.call(r,r.previous)||ac(r.events)?n(o):(e.enter("literalAutolink"),e.enter("literalAutolinkWww"),e.check(sC,e.attempt(yg,e.attempt(xg,a),n),n)(o))}function a(o){return e.exit("literalAutolinkWww"),e.exit("literalAutolink"),t(o)}}function pC(e,t,n){const r=this;let i="",a=!1;return o;function o(c){return(c===72||c===104)&&bg.call(r,r.previous)&&!ac(r.events)?(e.enter("literalAutolink"),e.enter("literalAutolinkHttp"),i+=String.fromCodePoint(c),e.consume(c),s):n(c)}function s(c){if(Ue(c)&&i.length<5)return i+=String.fromCodePoint(c),e.consume(c),s;if(c===58){const p=i.toLowerCase();if(p==="http"||p==="https")return e.consume(c),l}return n(c)}function l(c){return c===47?(e.consume(c),a?u:(a=!0,l)):n(c)}function u(c){return c===null||uo(c)||ce(c)||Qn(c)||Do(c)?n(c):e.attempt(yg,e.attempt(xg,d),n)(c)}function d(c){return e.exit("literalAutolinkHttp"),e.exit("literalAutolink"),t(c)}}function fC(e,t,n){let r=0;return i;function i(o){return(o===87||o===119)&&r<3?(r++,e.consume(o),i):o===46&&r===3?(e.consume(o),a):n(o)}function a(o){return o===null?n(o):t(o)}}function mC(e,t,n){let r,i,a;return o;function o(u){return u===46||u===95?e.check(_g,l,s)(u):u===null||ce(u)||Qn(u)||u!==45&&Do(u)?l(u):(a=!0,e.consume(u),o)}function s(u){return u===95?r=!0:(i=r,r=void 0),e.consume(u),o}function l(u){return i||r||!a?n(u):t(u)}}function hC(e,t){let n=0,r=0;return i;function i(o){return o===40?(n++,e.consume(o),i):o===41&&r<n?a(o):o===33||o===34||o===38||o===39||o===41||o===42||o===44||o===46||o===58||o===59||o===60||o===63||o===93||o===95||o===126?e.check(_g,t,a)(o):o===null||ce(o)||Qn(o)?t(o):(e.consume(o),i)}function a(o){return o===41&&r++,e.consume(o),i}}function gC(e,t,n){return r;function r(s){return s===33||s===34||s===39||s===41||s===42||s===44||s===46||s===58||s===59||s===63||s===95||s===126?(e.consume(s),r):s===38?(e.consume(s),a):s===93?(e.consume(s),i):s===60||s===null||ce(s)||Qn(s)?t(s):n(s)}function i(s){return s===null||s===40||s===91||ce(s)||Qn(s)?t(s):r(s)}function a(s){return Ue(s)?o(s):n(s)}function o(s){return s===59?(e.consume(s),r):Ue(s)?(e.consume(s),o):n(s)}}function yC(e,t,n){return r;function r(a){return e.consume(a),i}function i(a){return Me(a)?n(a):t(a)}}function kg(e){return e===null||e===40||e===42||e===95||e===91||e===93||e===126||ce(e)}function bg(e){return!Ue(e)}function Sg(e){return!(e===47||Vl(e))}function Vl(e){return e===43||e===45||e===46||e===95||Me(e)}function ac(e){let t=e.length,n=!1;for(;t--;){const r=e[t][1];if((r.type==="labelLink"||r.type==="labelImage")&&!r._balanced){n=!0;break}if(r._gfmAutolinkLiteralWalkedInto){n=!1;break}}return e.length>0&&!n&&(e[e.length-1][1]._gfmAutolinkLiteralWalkedInto=!0),n}const xC={tokenize:zC,partial:!0};function _C(){return{document:{91:{name:"gfmFootnoteDefinition",tokenize:bC,continuation:{tokenize:SC},exit:CC}},text:{91:{name:"gfmFootnoteCall",tokenize:kC},93:{name:"gfmPotentialFootnoteCall",add:"after",tokenize:vC,resolveTo:wC}}}}function vC(e,t,n){const r=this;let i=r.events.length;const a=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let o;for(;i--;){const l=r.events[i][1];if(l.type==="labelImage"){o=l;break}if(l.type==="gfmFootnoteCall"||l.type==="labelLink"||l.type==="label"||l.type==="image"||l.type==="link")break}return s;function s(l){if(!o||!o._balanced)return n(l);const u=Tt(r.sliceSerialize({start:o.end,end:r.now()}));return u.codePointAt(0)!==94||!a.includes(u.slice(1))?n(l):(e.enter("gfmFootnoteCallLabelMarker"),e.consume(l),e.exit("gfmFootnoteCallLabelMarker"),t(l))}}function wC(e,t){let n=e.length;for(;n--;)if(e[n][1].type==="labelImage"&&e[n][0]==="enter"){e[n][1];break}e[n+1][1].type="data",e[n+3][1].type="gfmFootnoteCallLabelMarker";const r={type:"gfmFootnoteCall",start:Object.assign({},e[n+3][1].start),end:Object.assign({},e[e.length-1][1].end)},i={type:"gfmFootnoteCallMarker",start:Object.assign({},e[n+3][1].end),end:Object.assign({},e[n+3][1].end)};i.end.column++,i.end.offset++,i.end._bufferIndex++;const a={type:"gfmFootnoteCallString",start:Object.assign({},i.end),end:Object.assign({},e[e.length-1][1].start)},o={type:"chunkString",contentType:"string",start:Object.assign({},a.start),end:Object.assign({},a.end)},s=[e[n+1],e[n+2],["enter",r,t],e[n+3],e[n+4],["enter",i,t],["exit",i,t],["enter",a,t],["enter",o,t],["exit",o,t],["exit",a,t],e[e.length-2],e[e.length-1],["exit",r,t]];return e.splice(n,e.length-n+1,...s),e}function kC(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a=0,o;return s;function s(c){return e.enter("gfmFootnoteCall"),e.enter("gfmFootnoteCallLabelMarker"),e.consume(c),e.exit("gfmFootnoteCallLabelMarker"),l}function l(c){return c!==94?n(c):(e.enter("gfmFootnoteCallMarker"),e.consume(c),e.exit("gfmFootnoteCallMarker"),e.enter("gfmFootnoteCallString"),e.enter("chunkString").contentType="string",u)}function u(c){if(a>999||c===93&&!o||c===null||c===91||ce(c))return n(c);if(c===93){e.exit("chunkString");const p=e.exit("gfmFootnoteCallString");return i.includes(Tt(r.sliceSerialize(p)))?(e.enter("gfmFootnoteCallLabelMarker"),e.consume(c),e.exit("gfmFootnoteCallLabelMarker"),e.exit("gfmFootnoteCall"),t):n(c)}return ce(c)||(o=!0),a++,e.consume(c),c===92?d:u}function d(c){return c===91||c===92||c===93?(e.consume(c),a++,u):u(c)}}function bC(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a,o=0,s;return l;function l(y){return e.enter("gfmFootnoteDefinition")._container=!0,e.enter("gfmFootnoteDefinitionLabel"),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionLabelMarker"),u}function u(y){return y===94?(e.enter("gfmFootnoteDefinitionMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionMarker"),e.enter("gfmFootnoteDefinitionLabelString"),e.enter("chunkString").contentType="string",d):n(y)}function d(y){if(o>999||y===93&&!s||y===null||y===91||ce(y))return n(y);if(y===93){e.exit("chunkString");const _=e.exit("gfmFootnoteDefinitionLabelString");return a=Tt(r.sliceSerialize(_)),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(y),e.exit("gfmFootnoteDefinitionLabelMarker"),e.exit("gfmFootnoteDefinitionLabel"),p}return ce(y)||(s=!0),o++,e.consume(y),y===92?c:d}function c(y){return y===91||y===92||y===93?(e.consume(y),o++,d):d(y)}function p(y){return y===58?(e.enter("definitionMarker"),e.consume(y),e.exit("definitionMarker"),i.includes(a)||i.push(a),ne(e,f,"gfmFootnoteDefinitionWhitespace")):n(y)}function f(y){return t(y)}}function SC(e,t,n){return e.check(Xi,t,e.attempt(xC,t,n))}function CC(e){e.exit("gfmFootnoteDefinition")}function zC(e,t,n){const r=this;return ne(e,i,"gfmFootnoteDefinitionIndent",5);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="gfmFootnoteDefinitionIndent"&&o[2].sliceSerialize(o[1],!0).length===4?t(a):n(a)}}function EC(e){let n=(e||{}).singleTilde;const r={name:"strikethrough",tokenize:a,resolveAll:i};return n==null&&(n=!0),{text:{126:r},insideSpan:{null:[r]},attentionMarkers:{null:[126]}};function i(o,s){let l=-1;for(;++l<o.length;)if(o[l][0]==="enter"&&o[l][1].type==="strikethroughSequenceTemporary"&&o[l][1]._close){let u=l;for(;u--;)if(o[u][0]==="exit"&&o[u][1].type==="strikethroughSequenceTemporary"&&o[u][1]._open&&o[l][1].end.offset-o[l][1].start.offset===o[u][1].end.offset-o[u][1].start.offset){o[l][1].type="strikethroughSequence",o[u][1].type="strikethroughSequence";const d={type:"strikethrough",start:Object.assign({},o[u][1].start),end:Object.assign({},o[l][1].end)},c={type:"strikethroughText",start:Object.assign({},o[u][1].end),end:Object.assign({},o[l][1].start)},p=[["enter",d,s],["enter",o[u][1],s],["exit",o[u][1],s],["enter",c,s]],f=s.parser.constructs.insideSpan.null;f&&ot(p,p.length,0,Fo(f,o.slice(u+1,l),s)),ot(p,p.length,0,[["exit",c,s],["enter",o[l][1],s],["exit",o[l][1],s],["exit",d,s]]),ot(o,u-1,l-u+3,p),l=u+p.length-2;break}}for(l=-1;++l<o.length;)o[l][1].type==="strikethroughSequenceTemporary"&&(o[l][1].type="data");return o}function a(o,s,l){const u=this.previous,d=this.events;let c=0;return p;function p(y){return u===126&&d[d.length-1][1].type!=="characterEscape"?l(y):(o.enter("strikethroughSequenceTemporary"),f(y))}function f(y){const _=Tr(u);if(y===126)return c>1?l(y):(o.consume(y),c++,f);if(c<2&&!n)return l(y);const S=o.exit("strikethroughSequenceTemporary"),m=Tr(y);return S._open=!m||m===2&&!!_,S._close=!_||_===2&&!!m,s(y)}}}class PC{constructor(){this.map=[]}add(t,n,r){NC(this,t,n,r)}consume(t){if(this.map.sort(function(a,o){return a[0]-o[0]}),this.map.length===0)return;let n=this.map.length;const r=[];for(;n>0;)n-=1,r.push(t.slice(this.map[n][0]+this.map[n][1]),this.map[n][2]),t.length=this.map[n][0];r.push(t.slice()),t.length=0;let i=r.pop();for(;i;){for(const a of i)t.push(a);i=r.pop()}this.map.length=0}}function NC(e,t,n,r){let i=0;if(!(n===0&&r.length===0)){for(;i<e.map.length;){if(e.map[i][0]===t){e.map[i][1]+=n,e.map[i][2].push(...r);return}i+=1}e.map.push([t,n,r])}}function TC(e,t){let n=!1;const r=[];for(;t<e.length;){const i=e[t];if(n){if(i[0]==="enter")i[1].type==="tableContent"&&r.push(e[t+1][1].type==="tableDelimiterMarker"?"left":"none");else if(i[1].type==="tableContent"){if(e[t-1][1].type==="tableDelimiterMarker"){const a=r.length-1;r[a]=r[a]==="left"?"center":"right"}}else if(i[1].type==="tableDelimiterRow")break}else i[0]==="enter"&&i[1].type==="tableDelimiterRow"&&(n=!0);t+=1}return r}function LC(){return{flow:{null:{name:"table",tokenize:AC,resolveAll:RC}}}}function AC(e,t,n){const r=this;let i=0,a=0,o;return s;function s(k){let I=r.events.length-1;for(;I>-1;){const te=r.events[I][1].type;if(te==="lineEnding"||te==="linePrefix")I--;else break}const W=I>-1?r.events[I][1].type:null,K=W==="tableHead"||W==="tableRow"?b:l;return K===b&&r.parser.lazy[r.now().line]?n(k):K(k)}function l(k){return e.enter("tableHead"),e.enter("tableRow"),u(k)}function u(k){return k===124||(o=!0,a+=1),d(k)}function d(k){return k===null?n(k):Q(k)?a>1?(a=0,r.interrupt=!0,e.exit("tableRow"),e.enter("lineEnding"),e.consume(k),e.exit("lineEnding"),f):n(k):Z(k)?ne(e,d,"whitespace")(k):(a+=1,o&&(o=!1,i+=1),k===124?(e.enter("tableCellDivider"),e.consume(k),e.exit("tableCellDivider"),o=!0,d):(e.enter("data"),c(k)))}function c(k){return k===null||k===124||ce(k)?(e.exit("data"),d(k)):(e.consume(k),k===92?p:c)}function p(k){return k===92||k===124?(e.consume(k),c):c(k)}function f(k){return r.interrupt=!1,r.parser.lazy[r.now().line]?n(k):(e.enter("tableDelimiterRow"),o=!1,Z(k)?ne(e,y,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(k):y(k))}function y(k){return k===45||k===58?S(k):k===124?(o=!0,e.enter("tableCellDivider"),e.consume(k),e.exit("tableCellDivider"),_):E(k)}function _(k){return Z(k)?ne(e,S,"whitespace")(k):S(k)}function S(k){return k===58?(a+=1,o=!0,e.enter("tableDelimiterMarker"),e.consume(k),e.exit("tableDelimiterMarker"),m):k===45?(a+=1,m(k)):k===null||Q(k)?C(k):E(k)}function m(k){return k===45?(e.enter("tableDelimiterFiller"),h(k)):E(k)}function h(k){return k===45?(e.consume(k),h):k===58?(o=!0,e.exit("tableDelimiterFiller"),e.enter("tableDelimiterMarker"),e.consume(k),e.exit("tableDelimiterMarker"),g):(e.exit("tableDelimiterFiller"),g(k))}function g(k){return Z(k)?ne(e,C,"whitespace")(k):C(k)}function C(k){return k===124?y(k):k===null||Q(k)?!o||i!==a?E(k):(e.exit("tableDelimiterRow"),e.exit("tableHead"),t(k)):E(k)}function E(k){return n(k)}function b(k){return e.enter("tableRow"),P(k)}function P(k){return k===124?(e.enter("tableCellDivider"),e.consume(k),e.exit("tableCellDivider"),P):k===null||Q(k)?(e.exit("tableRow"),t(k)):Z(k)?ne(e,P,"whitespace")(k):(e.enter("data"),T(k))}function T(k){return k===null||k===124||ce(k)?(e.exit("data"),P(k)):(e.consume(k),k===92?D:T)}function D(k){return k===92||k===124?(e.consume(k),T):T(k)}}function RC(e,t){let n=-1,r=!0,i=0,a=[0,0,0,0],o=[0,0,0,0],s=!1,l=0,u,d,c;const p=new PC;for(;++n<e.length;){const f=e[n],y=f[1];f[0]==="enter"?y.type==="tableHead"?(s=!1,l!==0&&(Ep(p,t,l,u,d),d=void 0,l=0),u={type:"table",start:Object.assign({},y.start),end:Object.assign({},y.end)},p.add(n,0,[["enter",u,t]])):y.type==="tableRow"||y.type==="tableDelimiterRow"?(r=!0,c=void 0,a=[0,0,0,0],o=[0,n+1,0,0],s&&(s=!1,d={type:"tableBody",start:Object.assign({},y.start),end:Object.assign({},y.end)},p.add(n,0,[["enter",d,t]])),i=y.type==="tableDelimiterRow"?2:d?3:1):i&&(y.type==="data"||y.type==="tableDelimiterMarker"||y.type==="tableDelimiterFiller")?(r=!1,o[2]===0&&(a[1]!==0&&(o[0]=o[1],c=wa(p,t,a,i,void 0,c),a=[0,0,0,0]),o[2]=n)):y.type==="tableCellDivider"&&(r?r=!1:(a[1]!==0&&(o[0]=o[1],c=wa(p,t,a,i,void 0,c)),a=o,o=[a[1],n,0,0])):y.type==="tableHead"?(s=!0,l=n):y.type==="tableRow"||y.type==="tableDelimiterRow"?(l=n,a[1]!==0?(o[0]=o[1],c=wa(p,t,a,i,n,c)):o[1]!==0&&(c=wa(p,t,o,i,n,c)),i=0):i&&(y.type==="data"||y.type==="tableDelimiterMarker"||y.type==="tableDelimiterFiller")&&(o[3]=n)}for(l!==0&&Ep(p,t,l,u,d),p.consume(t.events),n=-1;++n<t.events.length;){const f=t.events[n];f[0]==="enter"&&f[1].type==="table"&&(f[1]._align=TC(t.events,n))}return e}function wa(e,t,n,r,i,a){const o=r===1?"tableHeader":r===2?"tableDelimiter":"tableData",s="tableContent";n[0]!==0&&(a.end=Object.assign({},er(t.events,n[0])),e.add(n[0],0,[["exit",a,t]]));const l=er(t.events,n[1]);if(a={type:o,start:Object.assign({},l),end:Object.assign({},l)},e.add(n[1],0,[["enter",a,t]]),n[2]!==0){const u=er(t.events,n[2]),d=er(t.events,n[3]),c={type:s,start:Object.assign({},u),end:Object.assign({},d)};if(e.add(n[2],0,[["enter",c,t]]),r!==2){const p=t.events[n[2]],f=t.events[n[3]];if(p[1].end=Object.assign({},f[1].end),p[1].type="chunkText",p[1].contentType="text",n[3]>n[2]+1){const y=n[2]+1,_=n[3]-n[2]-1;e.add(y,_,[])}}e.add(n[3]+1,0,[["exit",c,t]])}return i!==void 0&&(a.end=Object.assign({},er(t.events,i)),e.add(i,0,[["exit",a,t]]),a=void 0),a}function Ep(e,t,n,r,i){const a=[],o=er(t.events,n);i&&(i.end=Object.assign({},o),a.push(["exit",i,t])),r.end=Object.assign({},o),a.push(["exit",r,t]),e.add(n+1,0,a)}function er(e,t){const n=e[t],r=n[0]==="enter"?"start":"end";return n[1][r]}const jC={name:"tasklistCheck",tokenize:DC};function IC(){return{text:{91:jC}}}function DC(e,t,n){const r=this;return i;function i(l){return r.previous!==null||!r._gfmTasklistFirstContentOfListItem?n(l):(e.enter("taskListCheck"),e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),a)}function a(l){return ce(l)?(e.enter("taskListCheckValueUnchecked"),e.consume(l),e.exit("taskListCheckValueUnchecked"),o):l===88||l===120?(e.enter("taskListCheckValueChecked"),e.consume(l),e.exit("taskListCheckValueChecked"),o):n(l)}function o(l){return l===93?(e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),e.exit("taskListCheck"),s):n(l)}function s(l){return Q(l)?t(l):Z(l)?e.check({tokenize:FC},t,n)(l):n(l)}}function FC(e,t,n){return ne(e,r,"whitespace");function r(i){return i===null?n(i):t(i)}}function MC(e){return yh([uC(),_C(),EC(e),LC(),IC()])}const OC={};function qC(e){const t=this,n=e||OC,r=t.data(),i=r.micromarkExtensions||(r.micromarkExtensions=[]),a=r.fromMarkdownExtensions||(r.fromMarkdownExtensions=[]),o=r.toMarkdownExtensions||(r.toMarkdownExtensions=[]);i.push(MC(n)),a.push(aC()),o.push(oC(n))}function WC({problem:e}){return x.jsxs("div",{className:"prose prose-sm max-w-none overflow-hidden",children:[x.jsxs("div",{className:"flex items-center gap-3 mb-4",children:[x.jsx("h1",{className:"text-xl font-bold text-gray-900 m-0",children:e.title}),x.jsx("span",{className:`px-2 py-0.5 rounded text-xs font-medium ${e.difficulty==="easy"?"bg-green-100 text-green-700":e.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:e.difficulty})]}),x.jsx(Oh,{remarkPlugins:[qC],components:{h2:({children:t})=>x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:t}),h3:({children:t})=>x.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:t}),p:({children:t})=>x.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:t}),code:({className:t,children:n})=>(t==null?void 0:t.includes("language-"))?x.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:x.jsx("code",{className:"text-sm text-gray-800",children:n})}):x.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm break-words",children:n}),ul:({children:t})=>x.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:t}),li:({children:t})=>x.jsx("li",{className:"text-gray-600",children:t}),table:({children:t})=>x.jsx("div",{className:"overflow-x-auto my-4",children:x.jsx("table",{className:"min-w-full border-collapse border border-gray-300 text-sm",children:t})}),thead:({children:t})=>x.jsx("thead",{className:"bg-gray-100",children:t}),tbody:({children:t})=>x.jsx("tbody",{className:"divide-y divide-gray-200",children:t}),tr:({children:t})=>x.jsx("tr",{children:t}),th:({children:t})=>x.jsx("th",{className:"border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700",children:t}),td:({children:t})=>x.jsx("td",{className:"border border-gray-300 px-3 py-2 text-gray-600",children:t})},children:e.description})]})}function BC({examples:e}){return x.jsxs("div",{className:"mt-6",children:[x.jsx("h3",{className:"text-base font-medium text-gray-900 mb-3",children:"Examples"}),x.jsx("div",{className:"space-y-4",children:e.map((t,n)=>x.jsx("div",{className:"bg-white rounded-lg p-4 border border-gray-200 shadow-sm",children:x.jsxs("div",{className:"space-y-2",children:[x.jsxs("div",{className:"overflow-hidden",children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Input:"}),x.jsx("pre",{className:"mt-1 text-sm text-primary-600 font-mono whitespace-pre-wrap break-all",children:t.input})]}),x.jsxs("div",{className:"overflow-hidden",children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Output:"}),x.jsx("pre",{className:"mt-1 text-sm text-green-600 font-mono whitespace-pre-wrap break-all",children:t.output})]}),t.explanation&&x.jsxs("div",{className:"pt-2 border-t border-gray-200",children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Explanation: "}),x.jsx("span",{className:"text-gray-600 text-sm",children:t.explanation})]})]})},n))})]})}function UC({hints:e,solution:t}){const[n,r]=z.useState(0),[i,a]=z.useState(!1),o=()=>{n<e.length&&r(l=>l+1)},s=()=>{a(!0)};return x.jsxs("div",{className:"mt-6 space-y-4",children:[x.jsxs("div",{children:[x.jsxs("div",{className:"flex items-center justify-between mb-3",children:[x.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Hints"}),n<e.length&&x.jsxs("button",{onClick:o,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:["Show Hint (",n,"/",e.length,")"]})]}),n>0?x.jsx("div",{className:"space-y-2",children:e.slice(0,n).map((l,u)=>x.jsx("div",{className:"bg-yellow-50 border border-yellow-200 rounded-lg p-3",children:x.jsxs("div",{className:"flex items-start gap-2",children:[x.jsx("span",{className:"text-yellow-600 flex-shrink-0",children:"💡"}),x.jsx("span",{className:"text-gray-700 text-sm break-words overflow-hidden",children:l})]})},u))}):x.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Show Hint" if you need help'})]}),x.jsxs("div",{children:[x.jsxs("div",{className:"flex items-center justify-between mb-3",children:[x.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Solution"}),!i&&n>=e.length&&x.jsx("button",{onClick:s,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:"Reveal Solution"})]}),i?x.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[x.jsx("div",{className:"px-4 py-2 bg-gray-100 border-b border-gray-200",children:x.jsx("span",{className:"text-sm text-gray-600",children:"Solution Code"})}),x.jsx("pre",{className:"p-4 text-sm text-gray-300 font-mono overflow-x-auto bg-gray-900",children:t})]}):n<e.length?x.jsx("div",{className:"text-gray-400 text-sm italic",children:"Reveal all hints first to unlock the solution"}):x.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Reveal Solution" to see the answer'})]})]})}function VC({results:e,isRunning:t}){if(t)return x.jsxs("div",{className:"flex items-center gap-3 p-4 bg-white rounded-lg border border-gray-200 shadow-sm",children:[x.jsx("div",{className:"w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"}),x.jsx("span",{className:"text-gray-600",children:"Running tests..."})]});if(e.length===0)return null;const n=e.filter(i=>i.passed).length,r=e.filter(i=>!i.hidden);return x.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[x.jsxs("div",{className:"px-4 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50",children:[x.jsx("span",{className:"font-medium text-gray-900",children:"Test Results"}),x.jsxs("span",{className:`text-sm font-medium ${n===e.length?"text-green-600":"text-yellow-600"}`,children:[n,"/",e.length," passed"]})]}),x.jsxs("div",{className:"divide-y divide-gray-200",children:[r.map(i=>x.jsxs("div",{className:"p-4",children:[x.jsxs("div",{className:"flex items-center gap-2 mb-2",children:[x.jsx("span",{className:`w-5 h-5 rounded-full flex items-center justify-center text-xs ${i.passed?"bg-green-100 text-green-600":"bg-red-100 text-red-600"}`,children:i.passed?"✓":"✗"}),x.jsxs("span",{className:"text-gray-700 text-sm font-medium",children:["Test ",i.id,": ",i.description]})]}),!i.passed&&x.jsxs("div",{className:"ml-7 space-y-1 text-sm",children:[x.jsxs("div",{children:[x.jsx("span",{className:"text-gray-500",children:"Expected: "}),x.jsx("span",{className:"text-green-600 font-mono",children:i.expected})]}),x.jsxs("div",{children:[x.jsx("span",{className:"text-gray-500",children:"Actual: "}),x.jsx("span",{className:"text-red-600 font-mono",children:i.actual})]})]})]},i.id)),e.some(i=>i.hidden)&&x.jsxs("div",{className:"p-4 text-gray-500 text-sm italic",children:["+ ",e.filter(i=>i.hidden).length," hidden test(s)",e.filter(i=>i.hidden&&i.passed).length===e.filter(i=>i.hidden).length?x.jsx("span",{className:"text-green-600 ml-2",children:"(all passed)"}):x.jsxs("span",{className:"text-red-600 ml-2",children:["(",e.filter(i=>i.hidden&&!i.passed).length," failed)"]})]})]})]})}function HC(){const{sectionId:e,problemId:t}=Gm(),{getProblemProgress:n,saveProblemCode:r,updateProblemStatus:i}=$i(),{isLoading:a,isReady:o,output:s,runTests:l,clearOutput:u}=Hh(),d=t?db(t):void 0,c=Nr.find(T=>T.id===e),[p,f]=z.useState(""),[y,_]=z.useState([]),[S,m]=z.useState(!1);z.useEffect(()=>{if(d&&e){const T=n(e,d.id);f(T.code||d.starterCode)}},[d,e,n]),z.useEffect(()=>{if(p&&d&&e&&p!==d.starterCode){const T=setTimeout(()=>{r(e,d.id,p)},1e3);return()=>clearTimeout(T)}},[p,d,e,r]);const h=z.useCallback(T=>{f(T)},[]),g=z.useCallback(()=>{d&&(f(d.starterCode),_([]),u())},[d,u]),C=T=>{const D=[...T.matchAll(/def\s+(\w+)\s*\(/g)];return D.length>0?D[D.length-1][1]:"solution"},E=z.useCallback(async()=>{if(!(!d||!o||!e)){m(!0),_([]);try{const T=C(d.starterCode),D=await l(p,d.testCases,T);_(D),D.every(I=>I.passed)&&i(e,d.id,"completed")}finally{m(!1)}}},[d,o,e,p,l,i]);if(!d||!c)return x.jsxs("div",{className:"text-center py-12",children:[x.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Problem Not Found"}),x.jsx(qt,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const b=c.problems.indexOf(d.id),P=c.problems[b+1];return x.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[x.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[x.jsxs("div",{className:"flex items-center gap-3",children:[x.jsxs(qt,{to:`/section/${e}`,className:"text-gray-500 hover:text-gray-900 transition-colors",children:["← ",c.title]}),x.jsx("span",{className:"text-gray-300",children:"/"}),x.jsx("span",{className:"text-gray-900 font-medium",children:d.title})]}),x.jsxs("div",{className:"flex items-center gap-3",children:[!o&&x.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),a?"Loading Python...":"Python ready"]}),o&&x.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),x.jsxs(Di,{className:"flex flex-1 overflow-hidden",sizes:[40,60],minSize:300,gutterSize:8,gutterAlign:"center",children:[x.jsxs("div",{className:"problem-panel overflow-y-auto overflow-x-hidden p-6 bg-gray-50 min-w-0",children:[x.jsx(WC,{problem:d}),x.jsx(BC,{examples:d.examples}),x.jsx(UC,{hints:d.hints,solution:d.solution})]}),x.jsxs("div",{className:"flex flex-col overflow-hidden bg-white",children:[x.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("button",{onClick:E,disabled:!o||S,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:S?"Running...":"Run Tests"}),x.jsx("button",{onClick:g,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"})]}),y.length>0&&y.every(T=>T.passed)&&P&&x.jsx(qt,{to:`/problem/${e}/${P}`,className:"px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors",children:"Next Problem →"})]}),x.jsxs(Di,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[x.jsx("div",{className:"h-full overflow-hidden p-4",children:x.jsx(eg,{value:p,onChange:h,height:"100%"})}),x.jsxs("div",{className:"overflow-auto p-4 space-y-4 bg-gray-50",children:[x.jsx(VC,{results:y,isRunning:S}),x.jsx(tg,{output:s,isLoading:S})]})]})]})]})]})}const Pp="ml-interview-scratchpad",Np=`import numpy as np

# Try out your code here!
# NumPy is already loaded and ready to use.

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
`;function $C(){const{isLoading:e,isReady:t,output:n,runCode:r,clearOutput:i}=Hh(),[a,o]=z.useState(""),[s,l]=z.useState(!1);z.useEffect(()=>{const p=localStorage.getItem(Pp);o(p||Np)},[]),z.useEffect(()=>{if(a){const p=setTimeout(()=>{localStorage.setItem(Pp,a)},1e3);return()=>clearTimeout(p)}},[a]);const u=z.useCallback(p=>{o(p)},[]),d=z.useCallback(()=>{o(Np),i()},[i]),c=z.useCallback(async()=>{if(t){l(!0),i();try{await r(a)}finally{l(!1)}}},[t,a,r,i]);return x.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[x.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[x.jsxs("div",{className:"flex items-center gap-3",children:[x.jsx(qt,{to:"/",className:"text-gray-500 hover:text-gray-900 transition-colors",children:"Home"}),x.jsx("span",{className:"text-gray-300",children:"/"}),x.jsx("span",{className:"text-gray-900 font-medium",children:"Python Scratchpad"})]}),x.jsxs("div",{className:"flex items-center gap-3",children:[!t&&x.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),e?"Loading Python...":"Python ready"]}),t&&x.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),x.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white",children:[x.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("button",{onClick:c,disabled:!t||s,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:s?"Running...":"Run Code"}),x.jsx("button",{onClick:d,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"}),x.jsx("button",{onClick:i,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Clear Output"})]}),x.jsx("span",{className:"text-gray-400 text-sm",children:"Code is auto-saved"})]}),x.jsxs(Di,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[x.jsx("div",{className:"h-full overflow-hidden p-4",children:x.jsx(eg,{value:a,onChange:u,height:"100%"})}),x.jsx("div",{className:"overflow-auto p-4 bg-gray-50",children:x.jsx(tg,{output:n,isLoading:s})})]})]})]})}function QC(){return x.jsx(m1,{children:x.jsxs(K0,{children:[x.jsx(ei,{path:"/",element:x.jsx(h1,{})}),x.jsx(ei,{path:"/section/:sectionId",element:x.jsx(fb,{})}),x.jsx(ei,{path:"/problem/:sectionId/:problemId",element:x.jsx(HC,{})}),x.jsx(ei,{path:"/scratchpad",element:x.jsx($C,{})})]})})}Ds.createRoot(document.getElementById("root")).render(x.jsx(Ft.StrictMode,{children:x.jsx(r1,{basename:"/ml-coding-lab",children:x.jsx(c1,{children:x.jsx(QC,{})})})}));
